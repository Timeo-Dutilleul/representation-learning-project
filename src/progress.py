from contextlib import contextmanager
import time


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(round(seconds), 60)
    return f"{minutes}m {remaining_seconds:02d}s"


@contextmanager
def progress_step(label: str):
    print(f"{label}...", end="", flush=True)
    started_at = time.perf_counter()
    try:
        yield
    except BaseException:
        elapsed = time.perf_counter() - started_at
        print(f" failed after {format_duration(elapsed)}", flush=True)
        raise
    else:
        elapsed = time.perf_counter() - started_at
        print(f" done in {format_duration(elapsed)}", flush=True)
