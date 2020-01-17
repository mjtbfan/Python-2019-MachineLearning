import math
import numpy as np
import scipy.stats as stats
import sklearn.utils as utl


def part_b(samples):
    return np.mean(samples)


def part_c(samples):
    return np.std(samples)


def part_d(data):
    log_mean = math.log(np.mean(data))
    return log_mean


def part_e(samples, mean, std):
    return stats.t.interval(0.95, len(samples) - 1, loc=mean, scale=std)


def main():
    x = [3, 5, 7, 18, 43, 85, 91, 98, 100, 130, 230, 487]
    sample_means = []
    for i in range(1000):
        sample_means.append(math.log(np.mean(utl.resample(x))))
    print("Part A - Log Means:", sample_means)
    print("Part B - Total Mean: %f" % part_b(sample_means))
    print("Part C - STD: %f" % part_c(sample_means))
    print("Part D - Data Log(Mean): %f" % part_d(x))
    print("       - Mean Difference: %f" % (part_b(sample_means) - part_d(x)))
    print("Part E - Confidence Interval:", part_e(sample_means, part_b(sample_means), part_c(sample_means)))


if __name__ == "__main__":
    main()