from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vep_comparisons.linear_probes.embedding_io import read_sites, save_embeddings
from vep_comparisons.protein.models import ESMCVariantModel, PositionRequest, SaProtVariantModel
from vep_comparisons.protein.windowing import aggregate_position_embeddings, sigmoid_window_weight
from vep_comparisons.rna.common import PreparedRequest, RnaCandidate, TranscriptAnnotation, prepare_requests
from vep_comparisons.rna.orthrus.model import OrthrusModel, build_six_track_input
from vep_comparisons.rna.rinalmo.model import RiNALMoModel, TOKEN_IDS, build_reference_tokens, select_transcript_window


def rna_request(sequence: str, target: int, transcript: str = 'T', strand: str = '+') -> PreparedRequest:
    annotation = TranscriptAnnotation(
        transcript, 'G', 'chr1', strand, sequence, ((100, 100 + len(sequence)),),
        tuple(i % 2 for i in range(len(sequence))), tuple(int(i == 2) for i in range(len(sequence))),
    )
    ref = sequence[target]
    alt = 'A' if ref != 'A' else 'C'
    if strand == '-':
        ref, alt = (base.translate(str.maketrans('ACGT', 'TGCA')) for base in (ref, alt))
    position = annotation.genomic_position(target)
    candidate = RnaCandidate(f'{transcript}:{position}', 'G', transcript, 'chr1', position,
                             position + 1, target + 1, ref, alt, 0)
    return prepare_requests([candidate], {transcript: annotation})[0]


class RinalmoHidden:
    def __init__(self) -> None:
        self.inputs: list[torch.Tensor] = []

    def __call__(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        assert not torch.is_grad_enabled()
        assert not (tokens == TOKEN_IDS['<mask>']).any()
        self.inputs.append(tokens.clone())
        positions = torch.arange(tokens.shape[1]).expand_as(tokens)
        counts = (tokens != TOKEN_IDS['<pad>']).sum(dim=1, keepdim=True).expand_as(tokens)
        return {'representation': torch.stack([tokens, positions, counts], dim=-1).float()}


def test_rinalmo_unmasked_crops_offsets_padding_and_reuse() -> None:
    requests = [rna_request('ACGT', 0), rna_request('ACGT', 3, strand='-'),
                rna_request('A' * 3000, 1500, 'long')]
    adapter = object.__new__(RiNALMoModel)
    adapter._torch = torch
    adapter.device = torch.device('cpu')
    adapter.model = RinalmoHidden()
    values = adapter.embed_requests(requests, batch_size=3)
    assert len(adapter.model.inputs) == 1
    assert adapter.model.inputs[0].shape == (2, 1024)  # one short crop reused
    for request in requests:
        tokens, window = build_reference_tokens(request)
        np.testing.assert_array_equal(values[request.key],
                                      [TOKEN_IDS[request.transcript_ref], window.target_token_index, len(tokens)])
    separate = adapter.embed_requests(requests, batch_size=1)
    for key in values:
        np.testing.assert_array_equal(values[key], separate[key])


@pytest.mark.parametrize('length,target', [(1023, 0), (1023, 1022), (3000, 2999)])
def test_rinalmo_reference_special_token_boundaries(length: int, target: int) -> None:
    request = rna_request('C' * length, target)
    tokens, window = build_reference_tokens(request)
    assert window == select_transcript_window(length, target)
    assert tokens[window.target_token_index] == TOKEN_IDS['C']
    assert (tokens[0] == TOKEN_IDS['<cls>']) == window.add_cls
    assert (tokens[-1] == TOKEN_IDS['<eos>']) == window.add_eos
    assert len(tokens) <= 1024


class OrthrusHidden:
    def __init__(self) -> None:
        self.inputs: list[torch.Tensor] = []

    def __call__(self, inputs: torch.Tensor, lengths: torch.Tensor, channel_last: bool) -> torch.Tensor:
        assert not torch.is_grad_enabled() and channel_last
        self.inputs.append(inputs.clone())
        for row, length in enumerate(lengths):
            assert inputs[row, length - 1, :4].sum() == 1
            assert not inputs[row, length:].any()
        return inputs.cumsum(dim=1)  # causal output, independent of trailing padding


def test_orthrus_reference_tracks_prefix_and_padding() -> None:
    requests = [rna_request('ACGTACGT', 2, strand='-'), rna_request('ACGTACGT', 6)]
    adapter = object.__new__(OrthrusModel)
    adapter._torch = torch
    adapter.device = torch.device('cpu')
    adapter.model = OrthrusHidden()
    values = adapter.embed_requests(requests, batch_size=2)
    for request in requests:
        index = request.candidate.target_index
        original = np.array(build_six_track_input(request.annotation, index))
        unmasked = np.array(build_six_track_input(request.annotation, index, mask_target=False))
        assert len(unmasked) == index + 1
        np.testing.assert_array_equal(original[:, 4:], unmasked[:, 4:])
        assert original[index, :4].sum() == 0 and unmasked[index, :4].sum() == 1
        np.testing.assert_array_equal(values[request.key], unmasked.sum(axis=0))
    separate = adapter.embed_requests(requests, batch_size=1)
    for key in values:
        np.testing.assert_array_equal(values[key], separate[key])


class ProteinTokenizer:
    def __init__(self, paired: bool) -> None:
        self.paired = paired
        self.sequences: list[str] = []

    def __call__(self, sequences: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
        self.sequences.extend(sequences)
        arrays = []
        for sequence in sequences:
            if self.paired:
                assert all(aa != '#' for aa in sequence[::2])
                ids = [ord(aa) * 100 + ord(structure) for aa, structure in zip(sequence[::2], sequence[1::2])]
            else:
                ids = [ord(aa) for aa in sequence]
            arrays.append([1, *ids, 2])
        tokens = torch.zeros((len(arrays), max(map(len, arrays))), dtype=torch.long)
        for row, ids in enumerate(arrays):
            tokens[row, :len(ids)] = torch.tensor(ids)
        return {'input_ids': tokens, 'attention_mask': (tokens != 0).long()}


class ProteinHidden:
    def __call__(self, tokens: torch.Tensor) -> SimpleNamespace:
        assert not torch.is_grad_enabled()
        positions = torch.arange(tokens.shape[1]).expand_as(tokens)
        return SimpleNamespace(embeddings=torch.stack([tokens, positions], dim=-1).float())

    def esm(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, return_dict: bool) -> SimpleNamespace:
        assert return_dict
        assert torch.equal(attention_mask, (input_ids != 0).long())
        return SimpleNamespace(last_hidden_state=self(input_ids).embeddings)


@pytest.mark.parametrize('family', ['esmc', 'saprot'])
def test_protein_reference_tokens_special_offsets_padding_and_window_reuse(family: str) -> None:
    adapter = object.__new__(ESMCVariantModel if family == 'esmc' else SaProtVariantModel)
    adapter.device = torch.device('cpu')
    adapter.tokenizer = ProteinTokenizer(paired=family == 'saprot')
    adapter.model = ProteinHidden()
    requests = [PositionRequest('T', 'ACDE', 11, 'pynw', 10),
                PositionRequest('T', 'ACDE', 14, 'pynw', 10),
                PositionRequest('short', 'FG', 2, 'rq')]
    result = adapter.embed_positions(requests, batch_size=2)
    assert len(adapter.tokenizer.sequences) == 2  # two focal positions share a forward pass
    for request in requests:
        index = request.local_position - 1
        expected = ord(request.sequence[index])
        if family == 'saprot':
            expected = expected * 100 + ord(request.structure_tokens[index])
        np.testing.assert_array_equal(result[request.key], [expected, index + 1])
    separate = adapter.embed_positions(requests, batch_size=1)
    for key in result:
        np.testing.assert_array_equal(result[key], separate[key])
    if family == 'saprot':
        assert adapter._combined_sequence(requests[0]) == '#pCyDnEw'  # scoring still masks


def test_vector_window_aggregation_uses_normalized_scoring_weights() -> None:
    requests = [PositionRequest('T', 'A' * 8, 5, window_start=0),
                PositionRequest('T', 'A' * 8, 5, window_start=4),
                PositionRequest('short', 'AA', 1)]
    embeddings = {requests[0].key: np.array([2, 4], dtype=np.float32),
                  requests[1].key: np.array([10, 8], dtype=np.float32),
                  requests[2].key: np.array([3, 5], dtype=np.float32)}
    result = aggregate_position_embeddings(requests, embeddings, {'T': 'A' * 12, 'short': 'AA'}, window_length=8)
    weights = [sigmoid_window_weight(r, protein_length=12, window_length=8) for r in requests[:2]]
    assert weights[0] != weights[1]
    expected = np.average([embeddings[r.key] for r in requests[:2]], axis=0, weights=weights)
    np.testing.assert_allclose(result['T', 5], expected)
    np.testing.assert_array_equal(result['short', 1], [3, 5])


def test_shards_and_saved_vectors_preserve_context_rows(tmp_path: Path) -> None:
    path = tmp_path / 'sites.tsv'
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['site_id', 'context_id', 'feature'], delimiter='\t')
        writer.writeheader()
        writer.writerows([{'site_id': str(i), 'context_id': f'c{i}', 'feature': transcript}
                          for i, transcript in enumerate(['T1', 'T2', 'T1', 'T3'])])
    first, second = (read_sites(path, 2, shard) for shard in range(2))
    assert [row['site_id'] for row in first] == ['0', '2', '3']
    assert [row['site_id'] for row in second] == ['1']
    vectors = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    output = save_embeddings(tmp_path, first, vectors, num_shards=2, shard_index=0)
    with np.load(output, allow_pickle=False) as saved:
        assert saved['context_id'].tolist() == ['c0', 'c2', 'c3']
        assert saved['site_id'].tolist() == ['0', '2', '3']
        np.testing.assert_array_equal(saved['embeddings'], vectors)
    with pytest.raises(AssertionError):
        save_embeddings(tmp_path, first, np.full((3, 2), np.nan), num_shards=2, shard_index=0)


@pytest.mark.parametrize('family', ['rinalmo', 'orthrus'])
def test_rna_extraction_command_maps_inputs_to_npz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family: str,
) -> None:
    from vep_comparisons.linear_probes import extract_rna
    from vep_comparisons.rna.rinalmo import model as rinalmo_module
    from vep_comparisons.rna.orthrus import model as orthrus_module

    adapter = object.__new__(RiNALMoModel if family == 'rinalmo' else OrthrusModel)
    adapter._torch = torch
    adapter.device = torch.device('cpu')
    adapter.model = RinalmoHidden() if family == 'rinalmo' else OrthrusHidden()
    module, name = (rinalmo_module, 'RiNALMoModel') if family == 'rinalmo' else (orthrus_module, 'OrthrusModel')
    monkeypatch.setattr(module, name, lambda *args: adapter)
    fasta, gff, sites = tmp_path / 'transcripts.fa', tmp_path / 'annotation.gff3', tmp_path / 'sites.tsv'
    fasta.write_text('>T\nACGT\n')
    gff.write_text(
        'chr1\tt\ttranscript\t101\t104\t.\t-\t.\tID=transcript:T;gene_id=G\n'
        'chr1\tt\texon\t101\t104\t.\t-\t.\tParent=transcript:T\n'
    )
    sites.write_text(
        'site_id\tcontext_id\tvariant\tgene\tfeature\tchromosome\tstart\tend\tref\talt\tcdna_position\n'
        'chr1:101\tcontext1\tv1\tG\tT\tchr1\t100\t101\tA\tC\t4\n'
        'chr1:104\tcontext2\tv2\tG\tT\tchr1\t103\t104\tT\tG\t1\n'
    )
    monkeypatch.setattr('sys.argv', ['extract_rna', '--sites', str(sites), '--model', family,
                                   '--transcripts', str(fasta), '--gff3', str(gff),
                                   '--checkpoint', str(tmp_path), '--output-dir', str(tmp_path),
                                   '--device', 'cpu', '--batch-size', '2'])
    extract_rna.main()
    with np.load(tmp_path / 'shard-00000-of-00001.npz') as saved:
        assert saved['site_id'].tolist() == ['chr1:101', 'chr1:104']
        assert saved['context_id'].tolist() == ['context1', 'context2']
        if family == 'rinalmo':
            np.testing.assert_array_equal(saved['embeddings'], [[TOKEN_IDS['T'], 4, 6], [TOKEN_IDS['A'], 1, 6]])
        else:
            np.testing.assert_array_equal(saved['embeddings'][:, :4], [[1, 1, 1, 1], [1, 0, 0, 0]])


@pytest.mark.parametrize('family', ['esmc', 'saprot'])
def test_protein_extraction_command_maps_sites_sharing_a_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family: str,
) -> None:
    from vep_comparisons.linear_probes import extract_protein

    adapter = object.__new__(ESMCVariantModel if family == 'esmc' else SaProtVariantModel)
    adapter.device = torch.device('cpu')
    adapter.tokenizer = ProteinTokenizer(paired=family == 'saprot')
    adapter.model = ProteinHidden()
    monkeypatch.setattr(extract_protein, 'load_variant_model', lambda *args, **kwargs: adapter)
    fasta, structures, sites = tmp_path / 'protein.fa', tmp_path / 'structures.fa', tmp_path / 'sites.tsv'
    fasta.write_text('>T\nACDE\n')
    structures.write_text('>T\npynw\n')
    sites.write_text(
        'site_id\tcontext_id\tvariant\tgene\tfeature\tprotein_position\tamino_acids\n'
        'chr1:1\tc1\tv1\tG\tT\t1\tA/C\n'
        'chr1:2\tc2\tv2\tG\tT\t1\tA/D\n'
        'chr1:3\tc3\tv3\tG\tT\t4\tE/F\n'
    )
    model = 'esmc-600m' if family == 'esmc' else 'saprot-650m'
    monkeypatch.setattr('sys.argv', ['extract_protein', '--sites', str(sites), '--model', model,
                                   '--sequences', str(fasta), '--structure-tokens', str(structures),
                                   '--model-dir', str(tmp_path), '--output-dir', str(tmp_path), '--device', 'cpu'])
    extract_protein.main()
    with np.load(tmp_path / 'shard-00000-of-00001.npz') as saved:
        assert saved['site_id'].tolist() == ['chr1:1', 'chr1:2', 'chr1:3']
        assert saved['context_id'].tolist() == ['c1', 'c2', 'c3']
        np.testing.assert_array_equal(saved['embeddings'][0], saved['embeddings'][1])
        np.testing.assert_array_equal(saved['embeddings'][:, 1], [1, 1, 4])
        assert len(adapter.tokenizer.sequences) == 1
