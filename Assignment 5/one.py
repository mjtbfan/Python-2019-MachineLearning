import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as sklm
import scipy.io as scio


def part_a(x, y):
    linear = sklm.SGDRegressor()
    lin_fit = linear.fit(x, y)
    lin_predict = lin_fit.predict(x)

    plt.figure()
    plt.scatter(x, y)
    plt.plot(x, lin_predict)
    plt.show()

    print('\nPart A - Weights:', lin_fit.coef_, 'Intercept:', lin_fit.intercept_, 'Iterations:', lin_fit.n_iter_)


def part_b(x, y):
    logist = sklm.SGDClassifier(loss='log')
    log_fit = logist.fit(x, y)
    log_predict = log_fit.predict(x)

    plt.figure()
    plt.scatter(x, y)
    plt.plot(np.sort(np.mean(x, axis=1), axis=None), np.sort(log_predict, axis=None))
    plt.show()

    print('\nPart B - Weights:', log_fit.coef_, 'Intercept:', log_fit.intercept_, 'Iterations:', log_fit.n_iter_)


def part_c(x, y):
    linear = sklm.LinearRegression()
    lin_fit = linear.fit(x, y)
    lin_predict = lin_fit.predict(x)

    plt.figure()
    plt.scatter(x, y)
    plt.plot(x, lin_predict)
    plt.show()

    print('\nPart C - Features:', lin_fit.coef_, 'Intercept:', lin_fit.intercept_)


def part_d(x, y):
    logist = sklm.LogisticRegression(solver='lbfgs')
    log_fit = logist.fit(x, y)
    log_predict = log_fit.predict(x)

    plt.figure()
    plt.scatter(x, y)
    plt.plot(np.sort(np.mean(x, axis=1), axis=None), np.sort(log_predict, axis=None))
    plt.show()

    print('\nPart D - Features:', log_fit.coef_, 'Intercept:', log_fit.intercept_)


def main():
    data = scio.loadmat('data/data_hw5_1.mat')
    data_frame = pd.DataFrame(data['xyz'], columns=['input', 'continuous', 'discrete'])

    reshape_x = data_frame['input'].values.reshape(-1, 1)
    reshape_y = data_frame['continuous'].values.reshape(-1, 1).ravel()
    reshape_z = data_frame['discrete'].values.reshape(-1, 1).ravel()
    part_a(reshape_x, reshape_y)
    part_b(reshape_x, reshape_z)
    part_c(reshape_x, reshape_y)
    part_d(reshape_x, reshape_z)


if __name__ == "__main__":
    main()