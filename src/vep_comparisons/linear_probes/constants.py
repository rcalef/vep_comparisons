"""Checkpoints included in the frozen-representation experiment."""

DEFAULT_MODELS = ('nt_100', 'nt_650', 'rinalmo', 'orthrus', 'esmc', 'saprot')
DIMENSIONS = {'nt_100': 768, 'nt_650': 1536, 'rinalmo': 1280,
              'orthrus': 512, 'esmc': 1152, 'saprot': 1280, 'evo2': 16384}
MODEL_LABELS = {'nt_100': 'NTv3 100M', 'nt_650': 'NTv3 650M', 'rinalmo': 'RiNALMo',
                'orthrus': 'Orthrus', 'esmc': 'ESM-C', 'saprot': 'SaProt', 'evo2': 'Evo2 7B'}
PROTEIN_MODELS = {'esmc', 'saprot'}
