import math
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as sklm
import scipy.io as scio


def part_a(x, y, n):
    logist = sklm.LogisticRegression(solver="lbfgs")
    log_fit = logist.fit(x, y)
    log_pred = log_fit.predict(x)
    log_score = log_fit.score(x, y)
    rss = np.sum((log_pred - y) ** 2)
    bic = n * math.log(rss / n) + 3 * math.log(n)

    print('\nPART A - Features:', log_fit.coef_, 'Intercept:', log_fit.intercept_)
    print('       - BIC:', bic, 'Score:', log_score)

    plt.figure()
    plt.scatter(x, (y, y))
    plt.plot(np.sort(np.mean(x, axis=1), axis=None), np.sort(log_pred, axis=None))
    plt.show()

def part_b(x, y, n):
    logist = sklm.LogisticRegression(solver="lbfgs")
    log_fit = logist.fit(x, y)
    log_pred = log_fit.predict(x)
    log_score = log_fit.score(x, y)
    rss = np.sum((log_pred - y) ** 2)
    bic = n * math.log(rss / n) + 6 * math.log(n)

    print('\nPART B - Features:', log_fit.coef_, 'Intercept:', log_fit.intercept_)
    print('       - BIC:', bic, 'Score:', log_score)

    plt.figure()
    plt.scatter(x, (y, y, y, y, y))
    plt.plot(np.sort(np.mean(x, axis=1), axis=None), np.sort(log_pred, axis=None))
    plt.show()

def part_c(x, y, n):
    score_list = []
    bic_list = []
    feat_list = []
    inter_list = []
    col_list = []
    for i in range(1, 6):
        for j in itertools.combinations(x.columns, i):
            logist = sklm.LogisticRegression(solver='lbfgs')
            log_fit = logist.fit(x[list(j)], y)
            log_pred = log_fit.predict(x[list(j)])
            rss = np.sum((log_pred - y) ** 2)
            score = log_fit.score(x[list(j)], y)
            score_list.append(score)
            bic = n * math.log(rss / n) + (len(j) + 1) * math.log(n)
            bic_list.append(bic)
            col_list.append(j)
            feat_list.append([log_fit.coef_])
            inter_list.append(log_fit.intercept_)

    final_frame = pd.DataFrame({'columns': col_list, 'score': score_list, 'BIC': bic_list, 'weight': feat_list, 'intercept': inter_list})
    final_frame.to_csv('3c.csv')
    pd.set_option('display.max_columns', 10)
    print('\nPART C')
    print(final_frame)


def main():
    data = scio.loadmat('data/data_hw5_3.mat')
    data_frame = pd.DataFrame(data['xz'], columns=['x0', 'x1', 'y'])

    reshape_x0 = data_frame['x0'].values.reshape(-1, 1)
    reshape_x1 = data_frame['x1'].values.reshape(-1, 1)
    reshape_y = data_frame['y'].values.reshape(-1, 1).ravel()

    reshape_x = np.concatenate((reshape_x0, reshape_x1), axis=1)

    n = reshape_x0.shape[0]

    part_a(reshape_x, reshape_y, n)

    reshape_x02 = np.square(reshape_x0)
    reshape_x12 = np.square(reshape_x1)
    reshape_x01 = np.multiply(reshape_x0, reshape_x1)

    reshape_x2 = np.concatenate((reshape_x, reshape_x02, reshape_x12, reshape_x01), axis=1)
    data_frame['x02'] = reshape_x02
    data_frame['x12'] = reshape_x12
    data_frame['x01'] = reshape_x01

    part_b(reshape_x2, reshape_y, n)
    part_c(data_frame.drop(columns='y', axis=1), reshape_y, n)


if __name__ == "__main__":
    main()