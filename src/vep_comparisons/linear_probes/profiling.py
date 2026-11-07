"""Optional extraction timing and GPU memory reporting in the job log."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter


@contextmanager
def profile_extraction(enabled: bool, device: str, sites: int) -> Iterator[None]:
    if not enabled:
        yield
        return
    import torch

    gpu = torch.device(device)
    torch.cuda.synchronize(gpu)
    torch.cuda.reset_peak_memory_stats(gpu)
    started = perf_counter()
    yield
    torch.cuda.synchronize(gpu)
    seconds = perf_counter() - started
    print(
        f'PILOT gpu={torch.cuda.get_device_name(gpu)!r} sites={sites} '
        f'extraction_seconds={seconds:.3f} sites_per_second={sites / seconds:.3f} '
        f'peak_allocated_gib={torch.cuda.max_memory_allocated(gpu) / 2**30:.3f} '
        f'peak_reserved_gib={torch.cuda.max_memory_reserved(gpu) / 2**30:.3f}',
        flush=True,
    )
