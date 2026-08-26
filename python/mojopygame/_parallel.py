from concurrent.futures import ThreadPoolExecutor


_WORKERS = 8
_pool = ThreadPoolExecutor(max_workers=_WORKERS, thread_name_prefix="mojopygame")


def ranges(size, function):
    workers = min(_WORKERS, size)
    chunk = (size + workers - 1) // workers
    starts = range(0, size, chunk)
    futures = [
        _pool.submit(function, start, min(start + chunk, size)) for start in starts
    ]
    return [future.result() for future in futures]
