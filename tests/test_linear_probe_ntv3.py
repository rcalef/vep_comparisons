from types import SimpleNamespace

import numpy as np
import polars as pl
import torch

from vep_comparisons.dna.ntv3 import EXPECTED_VOCABULARY, NTv3Model
from vep_comparisons.dna.windows import ScoringContext
from vep_comparisons.linear_probes.extract_ntv3 import selected_sites


class Reference:
    references = ["chr1"]
    sequence = "ACGTAC"

    def fetch(self, chromosome: str, start: int, end: int) -> str:
        return self.sequence[start:end]

    def get_reference_length(self, chromosome: str) -> int:
        return len(self.sequence)


class Tokenizer:
    def __call__(self, sequences: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
        assert kwargs == {"add_special_tokens": False, "padding": False, "return_tensors": "pt"}
        return {"input_ids": torch.tensor([[EXPECTED_VOCABULARY[b] for b in s] for s in sequences])}


class Model:
    def __call__(self, input_ids: torch.Tensor, output_hidden_states: bool) -> SimpleNamespace:
        assert output_hidden_states
        assert not torch.is_grad_enabled()
        assert not (input_ids == EXPECTED_VOCABULARY["<mask>"]).any()
        # Encode both base identity and absolute position to detect off-by-one RC errors.
        positions = torch.arange(input_ids.shape[1]).expand_as(input_ids)
        hidden = torch.stack([input_ids, positions], dim=-1).float()
        return SimpleNamespace(hidden_states=(hidden * -100, hidden))


def test_unmasked_final_tensor_and_padded_reverse_complement() -> None:
    adapter = object.__new__(NTv3Model)
    adapter.tokenizer = Tokenizer()
    adapter.device = torch.device("cpu")
    adapter.model = Model()
    np.testing.assert_array_equal(adapter._embed_batch(["ACGT"], [1]), [[8, 1]])
    # start=-3 pads the left, target=4 corresponds to reference C at genomic index 1.
    contexts = [
        ScoringContext("chr1", -3, 8, 4, "C", "T"),
        ScoringContext("chr1", 1, 8, 4, "C", "A"),
    ]
    vectors = adapter.embed_contexts(contexts, reference=Reference(), batch_size=1)
    # Forward C=8 and reverse G=9; target indices 4 and 3 average to 3.5.
    np.testing.assert_array_equal(vectors, [[8.5, 3.5], [8.5, 3.5]])
    np.testing.assert_array_equal(vectors, adapter.embed_contexts(contexts, reference=Reference(), batch_size=2))


def test_extraction_union_and_shards_preserve_identifiers() -> None:
    manifest = pl.DataFrame({
        "site_id": ["chr1:1", "chr1:1", "chr1:2", "chr1:3"],
        "context_id": ["a", "a", "b", "c"],
        "task": ["phylop", "gnocchi_z", "phylop", "phylop"],
    })
    first = selected_sites(manifest, 2, 0)
    second = selected_sites(manifest, 2, 1)
    assert first["site_id"].to_list() == ["chr1:1", "chr1:3"]
    assert second["site_id"].to_list() == ["chr1:2"]
    assert first["context_id"].to_list() == ["a", "c"]
