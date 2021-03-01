import argparse


def main():
    parser = argparse.ArgumentParser(description="Matrix multiplication in python")
    parser.add_argument('-t', '--threads',type=int, default=8, dest='threads', help="The number of threads the matrix multiplier will use")
    parser.add_argument('-s', '--size', type=int, default=400, dest='msize', help="The size of the matrix. e.x. -s 400 = 400x400 matrix")
    parser.add_argument()

if __name__ == '__main__':
    main()