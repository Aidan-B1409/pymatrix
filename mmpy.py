import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from time import perf_counter

row = 0

def main():
    timings = {
        'init time': 0.0,
        'mult time': 0.0,
        'out time': 0.0, 
        'mmults': 0.0,
        'size': 0.0, 
        'threads': 0.0,
    }
    parser = argparse.ArgumentParser(description="Matrix multiplication in python")
    parser.add_argument('-t', '--threads',type=int, default=8, dest='threads', help="The number of threads the matrix multiplier will use")
    parser.add_argument('-s', '--size', type=int, default=400, dest='msize', help="The size of the matrix. e.x. -s 400 = 400x400 matrix")
    parser.add_argument('-o', '--output_file', type=str, dest='out', help='destination of output file for the matrix. Defaults to stdout')

    args = parser.parse_args()

    # Create the initial matrices
    init_tic = perf_counter()

    left_matrix = np.array([[i + j * 2.0 for j in range(args.msize)] for i in range(args.msize)], dtype=np.float64)
    right_matrix = np.array([[i + j * 3.0 for j in range(args.msize)] for i in range(args.msize)], dtype=np.float64)
    result_matrix = np.zeros((args.msize, args.msize), dtype=np.float64)
    
    init_toc = perf_counter()
    timings['init time'] = init_toc - init_tic
    print("init done")
    
    # Spin up worker threads 
    run_tic = perf_counter()

    # lock = mp.Lock()
    # wthreads = [mp.Process(target = thread_func, args = (left_matrix, right_matrix, result_matrix, lock, tid))
    #     for tid in range(args.threads)]
    # print(len(wthreads))
    # [thread.start() for thread in wthreads]
    # [thread.join() for thread in wthreads]

    with ProcessPoolExecutor(max_workers=args.threads) as pool:
        futures = []
        for i in range(result_matrix.shape[0]):
            futures.append(pool.submit(mult_row, left_matrix[i], right_matrix[i], result_matrix.shape[0]))
        for i in range(result_matrix.shape[0]):
            result_matrix[i] = futures[i].result()

        # results = pool.map(mult_row, left_matrix, right_matrix, range(args.msize))


    run_toc = perf_counter()
    timings['mult time'] = run_toc - run_tic


    out_tic = perf_counter()
    if(args.out):
        with open(args.out, 'w') as f:
            for row in result_matrix:
                for element in row:
                    f.write(f'{element} ')
                f.write('\n')
    else:
        print(result_matrix)

    out_toc = perf_counter()
    timings['out time'] = out_toc - out_tic
    timings['mmults'] = (args.msize * args.msize) / timings['mult time']
    timings['threads'] = args.threads
    timings['size'] = args.msize

    print(timings)
    print(result_matrix.shape[0])



def mult_row(left_row, right_row, length):
    return [left_row[i] * right_row[i] for i in range(length)]


# def get_next_row(lock) -> int:
#     global row
#     t_row = 0;
#     lock.acquire() 
#     t_row = row
#     row += 1
#     lock.release()
#     return t_row


if __name__ == '__main__':
    main()