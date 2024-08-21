import concurrent
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

def thread_pool_wrapper(tasks, args, with_results=False):
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(task, arg): task for task, arg in zip(tasks, args)}

        if with_results:
            for future in as_completed(futures):
                result = future.result()
                yield result
        else:
            return futures

def process_pool_wrapper(tasks, args, with_results=False):
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(task, arg): task for task, arg in zip(tasks, args)}

        if with_results:
            for future in as_completed(futures):
                result = future.result()
                yield result
        else:
            return futures
