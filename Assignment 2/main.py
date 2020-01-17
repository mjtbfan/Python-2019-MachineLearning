# Nicholas Delli Carpini - CS 4342

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.io as scio
import sklearn as scil


def three_mle(x, y, n):
    a_mle = ((y * (n + 1)) / (n * x))
    b_mle = (y/n) - ((y * (n + 1)) / (n ** 2))
    mu_mle = a_mle * x + b_mle
    var_mle = math.sqrt((1/n) * ((x - mu_mle) ** 2))
    print("a=%f, b=%f, mu=%f, var=%f" % (a_mle, b_mle, mu_mle, var_mle))
    return a_mle, b_mle


def three_main(matrix):
    plt.figure()
    n = matrix.shape[0]
    temp_x = np.linspace(start=-5, stop=5, endpoint=True)
    y_sum = matrix['y'].sum()
    x_sum = matrix['x'].sum()
    a, b = three_mle(x_sum, y_sum, n)
    scatter = matrix.plot.scatter(x='x', y='y')
    line = plt.plot(temp_x, a * temp_x + b, 'g')
    plt.show()


def four_main(matrix):
    diseased = (matrix.loc[matrix['disease'] == 1.0]).reset_index()
    clean = (matrix.loc[matrix['disease'] == 0.0]).reset_index()

    plt.figure()
    plt.hist(diseased['continuous'], 30, label='Diseased')
    plt.hist(clean['continuous'], 30, label='Clean')
    plt.legend()
    plt.show()

    print(scil.linear_model)

    # positive = (matrix.loc[matrix['indicator'] == 1.0]).reset_index()
    # negative = (matrix.loc[matrix['indicator'] == 0.0]).reset_index()
    #
    #
    # false_pos = (clean.loc[clean['indicator'] == 1.0]).reset_index()
    # false_neg = (diseased.loc[diseased['indicator'] == 0.0]).reset_index()
    # true_pos = (diseased.loc[diseased['indicator'] == 1.0]).reset_index()
    # true_neg = (clean.loc[clean['indicator'] == 0.0]).reset_index()
    #
    # mu_pos = float(np.mean(true_pos['continuous']))
    # var_pos = float(np.var(true_pos['continuous']))
    # mu_neg = float(np.mean(true_neg['continuous']))
    # var_neg = float(np.var(true_neg['continuous']))
    # print("mu1=%f, var1=%f, mu2=%f, var2=%f" % (mu_pos, var_pos, mu_neg, var_neg))
    #
    # temp_x = np.linspace(start=-2, stop=4, endpoint=True)
    #
    # print("TP=%i, TN=%i, FP=%i, FN=%i" % (true_pos.shape[0], true_neg.shape[0], false_pos.shape[0], false_neg.shape[0]))
    #
    # pos_cont = 0
    # pos_dis = 0
    # neg_cont = 0
    # neg_dis = 0
    # for i, row in positive.iterrows():
    #     pos_cont += stats.norm.pdf(row['continuous'], mu_pos, var_pos)
    #     pos_dis += ((47/60) ** row['indicator']) * ((13/60) ** (1 - row['indicator']))
    # for i, row in negative.iterrows():
    #     neg_cont += stats.norm.pdf(row['continuous'], mu_neg, var_neg)
    #     neg_dis += ((25/40) ** row['indicator']) * ((15/40) ** (1 - row['indicator']))
    #
    # naive_pos = (pos_cont / 62) * (pos_dis / 62) * 0.6
    # naive_neg = (neg_cont / 38) * (neg_dis / 38) * 0.4
    #
    # pos_cont = (pos_cont / 62) * 0.6
    # pos_dis = (pos_dis / 62) * 0.6
    # neg_cont = (neg_cont / 38) * 0.4
    # neg_dis = (neg_dis / 38) * 0.4
    #
    # print("Discrete: %f, %f" % (pos_dis, neg_dis))
    # print("Continuous: %f, %f" % (pos_cont, neg_cont))
    # print("Naive: %f, %f" % (naive_pos, naive_neg))




def five_main():
    prior = 1/3
    mu_1, mu_2, mu_3 = [0, 0], [1, 1], [-1, 1]
    x_a, x_b = [-0.5, 0.5], [0.5, 0.5]
    sig_1 = [[0.7, 0], [0, 0.7]]
    sig_2 = [[0.8, 0.2], [0.2, 0.8]]
    sig_3 = [[0.8, 0.2], [0.2, 0.8]]

    print("\nx_a")
    print(stats.multivariate_normal.pdf(x_a, mu_1, sig_1) * prior)
    print(stats.multivariate_normal.pdf(x_a, mu_2, sig_2) * prior)
    print(stats.multivariate_normal.pdf(x_a, mu_3, sig_3) * prior)

    print("\nx_b")
    print(stats.multivariate_normal.pdf(x_b, mu_1, sig_1) * prior)
    print(stats.multivariate_normal.pdf(x_b, mu_2, sig_2) * prior)
    print(stats.multivariate_normal.pdf(x_b, mu_3, sig_3) * prior)

def seven_main():
    prior_1, prior_2, prior_3 = 0.4, 0.3, 0.3
    mu_1, mu_2, mu_3 = [1, -1], [-1, 1], [3, 3]
    sig_1 = [[2, 0], [0, 16]]
    sig_2 = [[1, 0], [0, 1]]
    sig_3 = [[4, 1], [1, 2]]
    x_1 = stats.multivariate_normal.rvs(mu_1, sig_1, size=int(1000 * prior_1))
    x_2 = stats.multivariate_normal.rvs(mu_2, sig_2, size=int(1000 * prior_2))
    x_3 = stats.multivariate_normal.rvs(mu_3, sig_3, size=int(1000 * prior_3))


    plt.figure()
    plt.scatter(x_1[:,0], x_1[:,1], label="X1")
    plt.scatter(x_2[:,0], x_2[:,1], label="X2")
    plt.scatter(x_3[:,0], x_3[:,1], label="X3")
    plt.legend()
    plt.show()

    x_big = np.vstack([x_1, x_2, x_3])

    mu_big = np.mean(x_big, axis=0)
    sig_big = np.cov(x_big, rowvar=False)

    print("mu = ", mu_big)
    print("var = ", sig_big)

    big_rand = stats.multivariate_normal.rvs(mu_big, sig_big, size=1000)

    plt.figure()
    plt.scatter(big_rand[:, 0], big_rand[:, 1])
    plt.show()


def main():
    three_list = np.array([2.0, 7.89, -3.6, -16.55, 2.2, 6.73, 4.9, 17.91, 1.5, 2.06, 3.1, 12.84, 2.2, 8.13, -0.9, -5.35, 0.9, 3.97, -2.4, -12.31])
    three_matrix = np.ndarray(shape=(int(len(three_list)/2), 2), buffer=three_list)
    three_frame = pd.DataFrame(three_matrix, columns=["x","y"])

    mat = scio.loadmat('assignment_2_problem_4')
    print(mat)
    mat_frame = pd.DataFrame(mat['xy'], columns=['indicator', 'continuous', 'disease'])

    # three_main(three_frame)
    four_main(mat_frame)
    # five_main()
    # seven_main()


if __name__ == "__main__":
    main()
