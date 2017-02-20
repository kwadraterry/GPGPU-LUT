import sys
import pylab as plt
import numpy as np


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    times = np.genfromtxt(args[0], delimiter=",")
    legends = ["Size, px", "CPU", "GPU", "multi-GPU"]
    plt.figure()
    plt.title("Running time of different histogram calculation methods on 'plasma' synthetic image.")

    for i in range(1, 4):
        plt.plot(times[:, 0], times[:, i], marker='^', label=legends[i])

    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.xlabel("Picture size, px (log)")
    plt.ylabel("Time, sec (log)")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
