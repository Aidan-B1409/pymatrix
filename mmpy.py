import argparse
import threading
import numpy as np

row = 0

def main():
    parser = argparse.ArgumentParser(description="Matrix multiplication in python")
    parser.add_argument('-t', '--threads',type=int, default=8, dest='threads', help="The number of threads the matrix multiplier will use")
    parser.add_argument('-s', '--size', type=int, default=400, dest='msize', help="The size of the matrix. e.x. -s 400 = 400x400 matrix")
    parser.add_argument('-o', '--output_file', type=str, dest='out', help='destination of output file for the matrix. Defaults to stdout')

    args = parser.parse_args()

    # Create the initial matrices
    left_matrix = np.array([[i + j * 2.0 for j in range(args.msize)] for i in range(args.msize)], dtype=np.float64)
    right_matrix = np.array([[i + j * 3.0 for j in range(args.msize)] for i in range(args.msize)], dtype=np.float64)
    result_matrix = np.zeros((args.msize, args.msize), dtype=np.float64)
    # Spin up worker threads 
    lock = threading.Lock()
    wthreads = [threading.Thread(target = thread_func, args = (left_matrix, right_matrix, result_matrix, lock, tid))
        for tid in range(args.threads)]
    [thread.start() for thread in wthreads]
    [thread.join() for thread in wthreads]

    if(args.out):
        with open(args.out, 'w') as f:
            for row in result_matrix:
                for element in row:
                    f.write(f'{element} ')
                f.write('\n')
    else:
        print(result_matrix)



def thread_func(l_matrix, r_matrix, result, lock, tid):
    t_row = get_next_row(lock)
    while t_row < result.shape[1]:
        for i in range(result.shape[0]):
            result[t_row][i] = l_matrix[t_row][i] * r_matrix[t_row][i]
        t_row = get_next_row(lock)


def get_next_row(lock) -> int:
    global row
    t_row = 0;
    lock.acquire() 
    t_row = row
    row += 1
    lock.release()
    return t_row


if __name__ == '__main__':
    main()