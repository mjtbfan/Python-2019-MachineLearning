import scipy.stats as stats
import scipy.optimize as opt


def b_func(x):
    return (0.3 * stats.norm.cdf(x, 1, 1)) + (0.7 * stats.norm.cdf(x, -1, 1)) - 1/2


def part_b():
    print("Median:", opt.fsolve(b_func, 0))


def main():
    part_b()


if __name__ == "__main__":
    main()