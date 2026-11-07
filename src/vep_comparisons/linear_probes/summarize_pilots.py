"""Summarize pilot logs and compare focal vectors across batch sizes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

import numpy as np

from .constants import DIMENSIONS


def summarize_trial(log: Path) -> dict[str, str | int | float]:
    model, case, batch = log.parts[-4:-1]
    text = log.read_text()
    row: dict[str, str | int | float] = {
        'model': model, 'case': case, 'batch_size': int(batch.removeprefix('batch-')),
        'status': 'failed' if 'Exit status:' in text else 'incomplete',
    }
    for line in text.splitlines():
        if line.startswith(('PILOT ', 'PILOT_OUTPUT ')):
            row.update({key: float(value) for key, value in re.findall(r'(\w+)=([0-9.]+)', line)})
        if 'Elapsed (wall clock) time' in line:
            elapsed = line.split('): ', 1)[1]
            row['wall_seconds'] = sum(float(v) * 60**i for i, v in enumerate(reversed(elapsed.split(':'))))
        if 'Maximum resident set size (kbytes):' in line:
            row['host_peak_gib'] = int(line.rsplit(':', 1)[1]) / 2**20
    if 'identifiers=matched finite=true' not in text:
        return row
    row['status'] = 'ok'
    row['dimension'] = int(row['dimension'])
    row['sites'] = int(row['sites'])
    row['bytes'] = int(row['bytes'])
    assert row['dimension'] == DIMENSIONS[model]
    with np.load(log.parent / 'shard-00000-of-00001.npz', allow_pickle=False) as data:
        vectors = data['embeddings'].astype(np.float64)
        assert np.isfinite(vectors).all()
        baseline = log.parent.parent / 'batch-1/shard-00000-of-00001.npz'
        if baseline.exists():
            with np.load(baseline, allow_pickle=False) as reference:
                assert np.array_equal(data['context_id'], reference['context_id'])
                ref = reference['embeddings'].astype(np.float64)
            cosine = (vectors * ref).sum(axis=1) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(ref, axis=1))
            row['min_cosine_vs_batch1'] = float(cosine.min())
            row['max_abs_diff_vs_batch1'] = float(np.abs(vectors - ref).max())
            row['max_relative_l2_vs_batch1'] = float((np.linalg.norm(vectors - ref, axis=1) / np.linalg.norm(ref, axis=1)).max())
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    rows = [summarize_trial(log) for log in sorted(args.run_dir.glob('*/*/batch-*/run.log'))]
    fields = ['model', 'case', 'batch_size', 'status', 'sites', 'dimension',
              'extraction_seconds', 'sites_per_second', 'peak_allocated_gib', 'peak_reserved_gib',
              'wall_seconds', 'host_peak_gib', 'bytes', 'min_cosine_vs_batch1',
              'max_abs_diff_vs_batch1', 'max_relative_l2_vs_batch1']
    with (args.run_dir / 'summary.tsv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['model']} {row['case']} batch={row['batch_size']}: {row['status']}")


if __name__ == '__main__':
    main()
