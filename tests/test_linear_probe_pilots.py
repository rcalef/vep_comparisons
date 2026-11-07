from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import polars as pl
from polars.testing import assert_frame_equal
import pytest
import torch

from vep_comparisons.linear_probes.prepare_pilots import longest_sites, random_sites
from vep_comparisons.linear_probes import profiling


def test_pilot_selection_is_deterministic_and_stresses_distinct_transcripts() -> None:
    sites = pl.DataFrame({'context_id': ['a', 'b', 'c', 'd', 'e'],
                          'feature': ['T1', 'T1', 'T2', 'T3', 'T4'],
                          'length': [100, 101, 10, 99, 20]})
    assert_frame_equal(random_sites(sites, 3, 12), random_sites(sites.reverse(), 3, 12))
    assert random_sites(sites, 20, 12).height == 5
    longest = longest_sites(sites, 'length', 2)
    assert longest['context_id'].to_list() == ['b', 'd']
    assert longest['feature'].n_unique() == 2
    assert_frame_equal(longest, longest_sites(sites.reverse(), 'length', 2))


def test_profile_synchronizes_and_reports_inference_metrics(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    calls = []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: calls.append('sync'))
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda device: calls.append('reset'))
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda device: 2 * 2**30)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda device: 3 * 2**30)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda device: 'Test GPU')
    times = iter([10.0, 14.0])
    monkeypatch.setattr(profiling, 'perf_counter', lambda: next(times))
    with profiling.profile_extraction(True, 'cuda', 8):
        calls.append('inference')
    assert calls == ['sync', 'reset', 'inference', 'sync']
    output = capsys.readouterr().out
    assert 'extraction_seconds=4.000 sites_per_second=2.000' in output
    assert 'peak_allocated_gib=2.000 peak_reserved_gib=3.000' in output
    with profiling.profile_extraction(False, 'cpu', 8):
        pass
    assert calls == ['sync', 'reset', 'inference', 'sync']


@pytest.mark.parametrize('task,model,module,cases,dtype', [
    (0, 'nt_100', 'extract_ntv3', 1, 'float32'),
    (1, 'nt_650', 'extract_ntv3', 1, 'float32'),
    (2, 'rinalmo', 'extract_rna', 2, 'bfloat16'),
    (3, 'orthrus', 'extract_rna', 2, 'float32'),
    (4, 'esmc', 'extract_protein', 2, 'float32'),
    (5, 'saprot', 'extract_protein', 2, 'float32'),
])
def test_gpu_array_dispatch_and_failure_continuation(
    tmp_path: Path, task: int, model: str, module: str, cases: int, dtype: str,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    fake_python = tmp_path / 'fake python'
    # Simulate a failed extraction (e.g. OOM) without loading models or GPUs.
    fake_python.write_text('''#!/usr/bin/env python3
import json, os, sys
if sys.argv[1] == '-m':
    with open(os.environ['PILOT_TEST_CALLS'], 'a') as handle:
        handle.write(json.dumps(sys.argv[1:]) + '\\n')
    sys.exit(7)
''')
    fake_python.chmod(0o755)
    for name in ('nvidia-smi', 'module'):
        stub = tmp_path / name
        stub.write_text('#!/bin/sh\nexit 0\n')
        stub.chmod(0o755)
    calls_file = tmp_path / 'calls.jsonl'
    env = {
        **os.environ,
        'PATH': f'{tmp_path}:{os.environ["PATH"]}',
        'PROBE_REPO_ROOT': str(repo), 'SLURM_ARRAY_TASK_ID': str(task),
        'SLURM_ARRAY_JOB_ID': 'pilot-test', 'PROBE_BATCH_SIZES': '1 2',
        'PROBE_PILOT_OUTPUT': str(tmp_path / 'outputs'),
        'PROBE_PILOT_INPUTS': str(tmp_path / 'input space'),
        'PROBE_MODEL_ROOT': str(tmp_path / 'weight space'),
        'PROBE_PROTEIN_MODEL_ROOT': str(tmp_path / 'protein space'),
        'PROBE_GCC_RUNTIME': str(tmp_path), 'PILOT_TEST_CALLS': str(calls_file),
        **{f'PROBE_{name}_PYTHON': str(fake_python) for name in ('NTV3', 'RINALMO', 'ORTHRUS', 'PROTEIN')},
    }
    result = subprocess.run(['bash', str(repo / 'scripts/pilot_linear_probes.sh')],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    calls = [json.loads(line) for line in calls_file.read_text().splitlines()]
    assert len(calls) == cases * 2  # a failed trial does not suppress the next one
    for args in calls:
        assert args[1] == f'vep_comparisons.linear_probes.{module}'
        assert args[args.index('--dtype') + 1] == dtype
        assert '--profile' in args
        assert f'/pilot-test/{model}/' in args[args.index('--output-dir') + 1]
        if module == 'extract_ntv3':
            size = '100' if task == 0 else '650'
            assert args[args.index('--model-dir') + 1].endswith(f'/NTv3_{size}M_pre')
            assert args[args.index('--manifest') + 1] == str(tmp_path / 'input space/random/ntv3.parquet')
        elif task == 5:
            assert '--structure-tokens' in args
        elif task == 4:
            assert '--structure-tokens' not in args
