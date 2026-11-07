from pathlib import Path

import numpy as np
import pytest

from vep_comparisons.linear_probes.summarize_pilots import summarize_trial


def test_pilot_summary_reports_timing_and_batch_differences(tmp_path: Path) -> None:
    for batch in (1, 2):
        trial = tmp_path / 'orthrus/random' / f'batch-{batch}'
        trial.mkdir(parents=True)
        np.savez(trial / 'shard-00000-of-00001.npz', context_id=['a', 'b'],
                 embeddings=np.ones((2, 512), dtype=np.float32) * (1 if batch == 1 else 1.02))
    log = trial / 'run.log'
    log.write_text(
        'PILOT sites=2 extraction_seconds=0.500 sites_per_second=4.000 '
        'peak_allocated_gib=1.000 peak_reserved_gib=2.000\n'
        '\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:02.50\n'
        '\tMaximum resident set size (kbytes): 1048576\n'
        '\tExit status: 0\n'
        'PILOT_OUTPUT sites=2 dimension=512 dtype=float32 bytes=4000 identifiers=matched finite=true\n'
    )
    row = summarize_trial(log)
    assert row['status'] == 'ok' and row['wall_seconds'] == 62.5
    assert row['host_peak_gib'] == 1
    assert row['min_cosine_vs_batch1'] == pytest.approx(1)
    assert row['max_relative_l2_vs_batch1'] == pytest.approx(0.02, abs=1e-7)
    assert row['max_abs_diff_vs_batch1'] == pytest.approx(0.02, abs=1e-7)


def test_failed_trial_is_not_reported_as_successful(tmp_path: Path) -> None:
    trial = tmp_path / 'rinalmo/long/batch-8'
    trial.mkdir(parents=True)
    log = trial / 'run.log'
    log.write_text('torch.OutOfMemoryError\n\tExit status: 1\n')
    assert summarize_trial(log)['status'] == 'failed'
