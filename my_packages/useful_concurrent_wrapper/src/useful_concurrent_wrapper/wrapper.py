import concurrent
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

def thread_pool_wrapper(tasks_with_args, with_results=False, max_workers=5):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        print('Running ThreadPoolExecutor')
        futures = {executor.submit(task, *args): task for task, args in tasks_with_args}

        if with_results:
            results = []

            for future in as_completed(futures):
                results.append(future.result())
            
            return results
        else:
            return futures

def process_pool_wrapper(tasks_with_args, with_results=False, max_workers=5):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        print('Running ProcessPoolExecutor')
        futures = {executor.submit(task, *args): task for task, args in tasks_with_args}

        if with_results:
            results = []

            for future in as_completed(futures):
                results.append(future.result())
            
            return results
        else:
            return futures

def main():
    pass

if __name__ == "__main__":
    main()
