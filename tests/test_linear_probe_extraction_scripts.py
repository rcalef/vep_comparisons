from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def extraction_env(tmp_path: Path) -> dict[str, str]:
    repo = Path(__file__).resolve().parents[1]
    data = tmp_path / 'data with spaces'
    files = [
        'model_weights/NTv3_100M_pre/model.safetensors',
        'model_weights/NTv3_650M_pre/model.safetensors',
        'model_weights/NTv3_100M_pre/config.json',
        'model_weights/NTv3_650M_pre/config.json',
        'model_weights/NTv3_100M_pre/vocab.json',
        'model_weights/NTv3_650M_pre/vocab.json',
        'model_weights/evo2_7b.pt',
        'model_weights/ntv3_base_model/__init__.py',
        'model_weights/ntv3_base_model/modeling_ntv3_pretrained.py',
        'model_weights/RiNALMo/rinalmo_giga_pretrained.pt',
        'model_weights/orthrus-mlm-6-track/config.json',
        'model_weights/orthrus-mlm-6-track/orthrus_hf.py',
        'model_weights/orthrus-mlm-6-track/model.safetensors',
        'protein_weights/esmc-600m-2024-12/data/weights/esmc_600m_2024_12_v0.pth',
        'protein_weights/SaProt_650M_AF2/pytorch_model.bin',
        'protein_weights/SaProt_650M_AF2/config.json',
        'protein_weights/SaProt_650M_AF2/vocab.txt',
        'assemblies/hg38/hg38.fa.gz', 'assemblies/hg38/hg38.fa.gz.fai',
        'assemblies/hg38/hg38.fa.gz.gzi',
        'assemblies/hg38/v50_gencode/gencode.v50.transcripts.fa.gz',
        'assemblies/hg38/v50_gencode/gencode.v50.annotation.gff3.gz',
        'assemblies/hg38/v50_gencode/gencode.v50.pc_translations.fa.gz',
        'structures.fa.bz2',
    ]
    for name in files:
        path = data / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    fake_python = tmp_path / 'fake python'
    fake_python.write_text('''#!/usr/bin/env python3
import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
if args[0] == '-c':
    sys.exit(0)
with open(os.environ['EXTRACTION_CALLS'], 'a') as handle:
    handle.write(json.dumps(args) + '\\n')
if args[0] != '-m':
    subprocess.run([sys.executable, *args], check=True)
else:
    if '--model' in args and args[args.index('--model') + 1] == 'rinalmo':
        from vep_comparisons.rna.rinalmo.model import validate_rinalmo_checkpoint
        validate_rinalmo_checkpoint(Path(args[args.index('--checkpoint') + 1]))
    sources = {}
    for flag in ('--reference', '--model-dir', '--model-code-dir', '--checkpoint',
                 '--transcripts', '--gff3', '--sequences', '--structure-tokens'):
        if flag in args:
            path = Path(args[args.index(flag) + 1])
            paths = path.rglob('*') if path.is_dir() else [path]
            sources[flag] = {str(p.relative_to(path) if path.is_dir() else p.name): p.read_text()
                             for p in paths if p.is_file()}
    if '--reference' in args:
        reference = args[args.index('--reference') + 1]
        for suffix in ('.fai', '.gzi'):
            path = Path(reference + suffix)
            sources['--reference'][path.name] = path.read_text()
    Path(os.environ['EXTRACTION_SOURCES']).write_text(json.dumps(sources))
    sys.exit(int(os.environ.get('EXTRACTION_EXIT', '0')))
''')
    fake_python.chmod(0o755)
    for name in ('nvidia-smi', 'module'):
        stub = tmp_path / name
        stub.write_text('#!/bin/sh\nexit 0\n')
        stub.chmod(0o755)
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    return {
        **os.environ, 'PATH': f'{tmp_path}:{os.environ["PATH"]}',
        'PROBE_REPO_ROOT': str(repo), 'SLURM_ARRAY_TASK_ID': '2',
        'PROBE_DATA_ROOT': str(data), 'PROBE_PROTEIN_MODEL_ROOT': str(data / 'protein_weights'),
        'PROBE_STRUCTURE_TOKENS': str(data / 'structures.fa.bz2'),
        'PROBE_EVO2_CHECKPOINT': str(data / 'model_weights/evo2_7b.pt'),
        'PROBE_PREPARED_DIR': str(tmp_path / 'prepared'),
        'PROBE_EMBEDDINGS_DIR': str(tmp_path / 'embeddings'),
        'PROBE_GCC_RUNTIME': str(tmp_path), 'SLURM_TMPDIR': str(scratch),
        'EXTRACTION_CALLS': str(tmp_path / 'calls.jsonl'),
        'EXTRACTION_SOURCES': str(tmp_path / 'sources.json'),
        **{f'PROBE_{name}_PYTHON': str(fake_python) for name in ('NTV3', 'RINALMO', 'ORTHRUS', 'PROTEIN', 'EVO2')},
    }


@pytest.mark.parametrize('model,module,batch,dtype', [
    ('nt_100', 'extract_ntv3', 2, 'float32'),
    ('nt_650', 'extract_ntv3', 2, 'float32'),
    ('rinalmo', 'extract_rna', 8, 'bfloat16'),
    ('orthrus', 'extract_rna', 2, 'float32'),
    ('esmc', 'extract_protein', 4, 'float32'),
    ('saprot', 'extract_protein', 4, 'float32'),
    ('evo2', 'extract_evo2', 2, 'bfloat16'),
])
def test_full_extraction_dispatch_and_local_staging(
    extraction_env: dict[str, str], model: str, module: str, batch: int, dtype: str,
) -> None:
    env = extraction_env
    script = Path(env['PROBE_REPO_ROOT']) / f'scripts/extract_linear_probes_{model}.sh'
    result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    calls = [json.loads(line) for line in Path(env['EXTRACTION_CALLS']).read_text().splitlines()]
    args = calls[-1]
    shards = '16' if model == 'evo2' else '4'
    assert args[:2] == ['-m', f'vep_comparisons.linear_probes.{module}']
    for flag, value in {'--batch-size': str(batch), '--dtype': dtype,
                        '--num-shards': shards, '--shard-index': '2', '--device': 'cuda',
                        '--output-dir': f'{env["PROBE_EMBEDDINGS_DIR"]}/{model}'}.items():
        assert args[args.index(flag) + 1] == value
    input_flag, input_file = ('--manifest', 'manifest.parquet') if model.startswith('nt_') or model == 'evo2' else (
        '--sites', 'inputs/protein.tsv' if model in ('esmc', 'saprot') else 'inputs/dna_rna.tsv')
    assert args[args.index(input_flag) + 1] == f'{env["PROBE_PREPARED_DIR"]}/{input_file}'
    assert '--profile' in args
    sources = json.loads(Path(env['EXTRACTION_SOURCES']).read_text())
    assert all(sources.values())
    assert all(args[args.index(flag) + 1].startswith(env['SLURM_TMPDIR']) for flag in sources)
    staged = {Path(call[1]).name for call in calls[:-1]}
    if model.startswith('nt_'):
        assert {'model.safetensors', 'hg38.fa.gz'} <= staged
        assert {'hg38.fa.gz.fai', 'hg38.fa.gz.gzi'} <= sources['--reference'].keys()
        assert 'vocab.json' in sources['--model-dir']
        assert '__init__.py' in sources['--model-code-dir']
        assert sources['--model-dir']['model.safetensors'] == f'model_weights/NTv3_{model[3:]}M_pre/model.safetensors'
    elif model == 'evo2':
        assert {'evo2_7b.pt', 'hg38.fa.gz'} <= staged
        assert sources['--checkpoint']['evo2_7b.pt'] == 'model_weights/evo2_7b.pt'
    elif model == 'orthrus':
        assert 'orthrus_hf.py' in sources['--checkpoint']
    elif model == 'rinalmo':
        from vep_comparisons.rna.rinalmo.model import WEIGHTS_FILENAME

        assert sources['--checkpoint'] == {
            WEIGHTS_FILENAME: f'model_weights/RiNALMo/{WEIGHTS_FILENAME}',
        }
    elif model == 'saprot':
        assert 'SaProt_650M_AF2/vocab.txt' in sources['--model-dir']
        assert '--structure-tokens' in args
    elif model == 'esmc':
        assert '--structure-tokens' not in args
    assert not list(Path(env['SLURM_TMPDIR']).iterdir())
    assert (Path(env['PROBE_EMBEDDINGS_DIR']) / model / f'shard-00002-of-{int(shards):05d}.log').exists()


def test_extraction_overrides_and_failure_exit(extraction_env: dict[str, str]) -> None:
    env = {**extraction_env, 'PROBE_STAGE_INPUTS': '0', 'PROBE_BATCH_SIZE': '1',
           'PROBE_NUM_SHARDS': '8', 'EXTRACTION_EXIT': '7'}
    script = Path(env['PROBE_REPO_ROOT']) / 'scripts/extract_linear_probes_orthrus.sh'
    result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
    assert result.returncode == 7
    calls = [json.loads(line) for line in Path(env['EXTRACTION_CALLS']).read_text().splitlines()]
    assert len(calls) == 1
    args = calls[0]
    assert args[args.index('--batch-size') + 1] == '1'
    assert args[args.index('--num-shards') + 1] == '8'
    assert args[args.index('--checkpoint') + 1].startswith(env['PROBE_DATA_ROOT'])
    assert not list(Path(env['SLURM_TMPDIR']).iterdir())


def test_spooled_sbatch_wrapper_finds_repository(extraction_env: dict[str, str], tmp_path: Path) -> None:
    env = extraction_env.copy()
    repo = env.pop('PROBE_REPO_ROOT')
    env.update(SLURM_SUBMIT_DIR=repo, PROBE_STAGE_INPUTS='0')
    spooled = tmp_path / 'slurm_script'
    spooled.write_text((Path(repo) / 'scripts/extract_linear_probes_nt_100.sh').read_text())
    result = subprocess.run(['bash', str(spooled)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_evo2_scoring_script_uses_unified_cli_and_sixteen_shards(extraction_env: dict[str, str]) -> None:
    from vep_comparisons.cli import parse_args

    env = {
        **extraction_env, 'EVO2_REPO_ROOT': extraction_env['PROBE_REPO_ROOT'],
        'EVO2_PYTHON': extraction_env['PROBE_EVO2_PYTHON'],
        'EVO2_CHECKPOINT': extraction_env['PROBE_EVO2_CHECKPOINT'],
        'EVO2_REFERENCE': f'{extraction_env["PROBE_DATA_ROOT"]}/assemblies/hg38/hg38.fa.gz',
        'EVO2_VARIANTS': 'variants.tsv',
    }
    script = Path(env['PROBE_REPO_ROOT']) / 'scripts/score_collated_variants_evo2.sh'
    result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    call = json.loads(Path(env['EXTRACTION_CALLS']).read_text().splitlines()[-1])
    assert call[:2] == ['-m', 'vep_comparisons.cli']
    args = parse_args(call[2:])
    assert args.model == 'evo2-7b' and args.num_shards == 16 and args.shard_index == 2
    assert args.batch_size == 2 and args.dtype == 'bfloat16'
    for name in ('score_collated_variants_evo2.sh', 'extract_linear_probes_evo2.sh'):
        text = (Path(env['PROBE_REPO_ROOT']) / 'scripts' / name).read_text()
        assert '#SBATCH --array=0-15\n' in text
