import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as sklm


def part_a(x1, x2, x3, y, n):
    x1_lin = sklm.LinearRegression()
    x1_fit = x1_lin.fit(x1, y)
    x1_pred = x1_fit.predict(x1)
    x1_rss = np.sum((x1_pred - y) ** 2)
    x1_bic = n * math.log(x1_rss / n) + 2 * math.log(n)
    print("EXAM1 RSS: %f | BIC: %f" % (x1_rss, x1_bic))

    x2_lin = sklm.LinearRegression()
    x2_fit = x2_lin.fit(x2, y)
    x2_pred = x2_fit.predict(x2)
    x2_rss = np.sum((x2_pred - y) ** 2)
    x2_bic = n * math.log(x2_rss / n) + 2 * math.log(n)
    print("EXAM2 RSS: %f | BIC: %f" % (x2_rss, x2_bic))

    x3_lin = sklm.LinearRegression()
    x3_fit = x3_lin.fit(x3, y)
    x3_pred = x3_fit.predict(x3)
    x3_rss = np.sum((x3_pred - y) ** 2)
    x3_bic = n * math.log(x3_rss / n) + 2 * math.log(n)
    print("EXAM3 RSS: %f | BIC: %f" % (x3_rss, x3_bic))

    plt.figure()
    plt.scatter(x1, y, color='blue')
    plt.plot(x1, x1_pred, color='red')
    plt.title("EXAM1")

    plt.figure()
    plt.scatter(x2, y, color='blue')
    plt.plot(x2, x2_pred, color='red')
    plt.title("EXAM2")

    plt.figure()
    plt.scatter(x3, y, color='blue')
    plt.plot(x3, x3_pred, color='red')
    plt.title("EXAM3")

    plt.show()


def part_b(x1, x2, x3, y, n):
    x1_lin = sklm.LinearRegression()
    x1_fit = x1_lin.fit(x1, y)
    x1_pred = x1_fit.predict(x1)
    x1_rss = np.sum((x1_pred - y) ** 2)
    x1_bic = n * math.log(x1_rss / n) + 3 * math.log(n)
    print("EXAM1&2 RSS: %f | BIC: %f" % (x1_rss, x1_bic))

    x2_lin = sklm.LinearRegression()
    x2_fit = x2_lin.fit(x2, y)
    x2_pred = x2_fit.predict(x2)
    x2_rss = np.sum((x2_pred - y) ** 2)
    x2_bic = n * math.log(x2_rss / n) + 3 * math.log(n)
    print("EXAM1&3 RSS: %f | BIC: %f" % (x2_rss, x2_bic))

    x3_lin = sklm.LinearRegression()
    x3_fit = x3_lin.fit(x3, y)
    x3_pred = x3_fit.predict(x3)
    x3_rss = np.sum((x3_pred - y) ** 2)
    x3_bic = n * math.log(x3_rss / n) + 3 * math.log(n)
    print("EXAM2&3 RSS: %f | BIC: %f" % (x3_rss, x3_bic))

    plt.figure()
    plt.scatter(x1, (y, y), color='blue')
    plt.plot(np.sort(np.mean(x1, axis=1), axis=None), np.sort(x1_pred, axis=None), color='red')
    plt.title("EXAM1&2")

    plt.figure()
    plt.scatter(x2, (y, y), color='blue')
    plt.plot(np.sort(np.mean(x2, axis=1), axis=None), np.sort(x2_pred, axis=None), color='red')
    plt.title("EXAM1&3")

    plt.figure()
    plt.scatter(x3, (y, y), color='blue')
    plt.plot(np.sort(np.mean(x3, axis=1), axis=None), np.sort(x3_pred, axis=None), color='red')
    plt.title("EXAM2&3")

    plt.show()


def part_c(x1, y, n):
    x1_lin = sklm.LinearRegression()
    x1_fit = x1_lin.fit(x1, y)
    x1_pred = x1_fit.predict(x1)
    x1_rss = np.sum((x1_pred - y) ** 2)
    x1_bic = n * math.log(x1_rss / n) + 4 * math.log(n)
    print("EXAM1&2&3 RSS: %f | BIC: %f" % (x1_rss, x1_bic))
    print("Mean Residual: %f" % np.mean((x1_pred - y)))

    plt.figure()
    plt.scatter(x1, (y, y, y), color='blue')
    plt.plot(np.sort(np.mean(x1, axis=1), axis=None), np.sort(x1_pred, axis=None), color='red')
    plt.title("EXAM1&2&3")
    plt.show()


def main():
    data = pd.read_excel('data.xls')

    reshape_x1 = data['EXAM1'].values.reshape(-1, 1)
    reshape_x2 = data['EXAM2'].values.reshape(-1, 1)
    reshape_x3 = data['EXAM3'].values.reshape(-1, 1)
    reshape_y = data['FINAL'].values.reshape(-1, 1).ravel()

    reshape_x1_x2 = np.append(reshape_x1, reshape_x2, axis=1)
    reshape_x1_x3 = np.append(reshape_x1, reshape_x3, axis=1)
    reshape_x2_x3 = np.append(reshape_x2, reshape_x3, axis=1)

    reshape_x1_x2_x3 = np.append(reshape_x1, reshape_x2_x3, axis=1)

    n = reshape_x1.shape[0]

    part_a(reshape_x1, reshape_x2, reshape_x3, reshape_y, n)
    part_b(reshape_x1_x2, reshape_x1_x3, reshape_x2_x3, reshape_y, n)
    part_c(reshape_x1_x2_x3, reshape_y, n)


if __name__ == "__main__":
    main()