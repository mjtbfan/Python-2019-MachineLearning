import math
import itertools
import numpy as np
import pandas as pd
import sklearn.linear_model as sklm
import sklearn.metrics as skmt
import scipy.io as scio
import mlxtend.feature_selection as mlfs


def part_a(x, y, n):
    linear = sklm.LinearRegression()
    lin_fit = linear.fit(x, y)
    lin_pred = lin_fit.predict(x)
    rss = np.sum((lin_pred - y) ** 2)
    bic = n * math.log(rss / n) + 6 * math.log(n)
    r2 = skmt.r2_score(y, lin_pred)

    print('\nPART A - Features:', lin_fit.coef_, 'Intercept:', lin_fit.intercept_)
    print('       - RSS:', rss, 'BIC:', bic, 'R2:', r2)


def part_b(x, y, n):
    rss_list = []
    bic_list = []
    feat_list = []
    inter_list = []
    col_list = []
    for i in range(1, 6):
        for j in itertools.combinations(x.columns, i):
            linear = sklm.LinearRegression()
            lin_fit = linear.fit(x[list(j)], y)
            lin_pred = lin_fit.predict(x[list(j)])
            rss = np.sum((lin_pred - y) ** 2)
            rss_list.append(rss)
            bic = n * math.log(rss / n) + (len(j) + 1) * math.log(n)
            bic_list.append(bic)
            col_list.append(j)
            feat_list.append([lin_fit.coef_])
            inter_list.append(lin_fit.intercept_)

    final_frame = pd.DataFrame({'columns': col_list, 'RSS': rss_list, 'BIC': bic_list, 'weight': feat_list, 'intercept': inter_list})
    final_frame.to_csv('2b.csv')
    pd.set_option('display.max_columns', 10)
    print('\nPART B')
    print(final_frame)


def part_c(x, y):
    linear = sklm.LinearRegression()
    sfs = mlfs.SequentialFeatureSelector(linear, k_features=5, floating=False, forward=True, cv=0)
    sfs_fit = sfs.fit(x, y)
    print('\nPART C')
    print(sfs_fit.subsets_)


def part_d(x, y):
    linear = sklm.LinearRegression()
    sfs = mlfs.SequentialFeatureSelector(linear, k_features=1, floating=False, forward=False, cv=0)
    sfs_fit = sfs.fit(x, y)
    print('\nPART D')
    print(sfs_fit.subsets_)


def part_e(x, y, n):
    ridge = sklm.Ridge()
    rid_fit = ridge.fit(x, y)
    rid_pred = rid_fit.predict(x)
    rss = np.sum((rid_pred - y) ** 2)
    bic = n * math.log(rss / n) + 6 * math.log(n)
    r2 = skmt.r2_score(y, rid_pred)

    print('\nPART E - Features:', rid_fit.coef_, 'Intercept:', rid_fit.intercept_)
    print('       - RSS:', rss, 'BIC:', bic, 'R2:', r2)


def part_f(x, y, n):
    lasso = sklm.Lasso()
    las_fit = lasso.fit(x, y)
    las_pred = las_fit.predict(x)
    rss = np.sum((las_pred - y) ** 2)
    bic = n * math.log(rss / n) + 3 * math.log(n)
    r2 = skmt.r2_score(y, las_pred)

    print('\nPART F - Features:', las_fit.coef_, 'Intercept:', las_fit.intercept_)
    print('       - RSS:', rss, 'BIC:', bic, 'R2:', r2)


def main():
    data = scio.loadmat('data/data_hw5_2.mat')
    data_frame = pd.DataFrame(data['xy'], columns=['x0', 'x1', 'x2', 'x3', 'x4', 'y'])

    reshape_x0 = data_frame['x0'].values.reshape(-1, 1)
    reshape_x1 = data_frame['x1'].values.reshape(-1, 1)
    reshape_x2 = data_frame['x2'].values.reshape(-1, 1)
    reshape_x3 = data_frame['x3'].values.reshape(-1, 1)
    reshape_x4 = data_frame['x4'].values.reshape(-1, 1)

    reshape_x = np.concatenate((reshape_x0, reshape_x1, reshape_x2, reshape_x3, reshape_x4), axis=1)
    reshape_y = data_frame['y'].values.reshape(-1, 1).ravel()

    n = reshape_x0.shape[0]

    part_a(reshape_x, reshape_y, n)
    part_b(data_frame.drop(columns='y', axis=1), reshape_y, n)
    part_c(data_frame.drop(columns='y', axis=1), reshape_y)
    part_d(data_frame.drop(columns='y', axis=1), reshape_y)
    part_e(reshape_x, reshape_y, n)
    part_f(reshape_x, reshape_y, n)


if __name__ == "__main__":
    main()