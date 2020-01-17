import sys
import math
import numpy as np
import pandas as pd
import scipy.stats as sci
import matplotlib.pyplot as plt
import sklearn.naive_bayes as nb
import sklearn.discriminant_analysis as da
import sklearn.metrics as met
import mlxtend.plotting as mlxplt

np.set_printoptions(threshold=sys.maxsize)


def five_main(data):
    class_1 = data[data['y'] == 1].reset_index()
    class_2 = data[data['y'] == 2].reset_index()

    # PART A

    prior_1 = len(class_1)/(len(data['x']))
    prior_2 = len(class_2)/(len(data['x']))

    mu_1, var_1 = sci.norm.fit(class_1['x'])
    mu_2, var_2 = sci.norm.fit(class_2['x'])

    print("Class 1 Norm: mu = %f | var = %f | prior = %f" % (mu_1, var_1, prior_1))
    print("Class 2 Norm: mu = %f | var = %f | prior = %f" % (mu_2, var_2, prior_2))

    # PART B

    linespace_1 = np.linspace(class_1['x'].min(), class_1['x'].max())
    linespace_2 = np.linspace(class_2['x'].min(), class_2['x'].max())

    plt.figure()
    plt.hist(class_1['x'], 30, density=True, label="Class 1")
    plt.hist(class_2['x'], 30, density=True, label="Class 2")
    plt.plot(linespace_1, sci.norm.pdf(linespace_1, mu_1, var_1), label="Norm Class 1")
    plt.plot(linespace_2, sci.norm.pdf(linespace_2, mu_2, var_2), label="Norm Class 2")
    plt.plot(linespace_1, (3/4 * sci.norm.pdf(linespace_1, -3, 1)) + (1/4 * sci.norm.pdf(linespace_1, 7, 0.316)), label="Class 1 F")
    plt.legend()
    plt.show()

    # PART C

    reshape_y = data['y'].values.reshape(-1, 1).ravel()
    reshape_x = data['x'].values.reshape(-1, 1)

    norm_classifier = nb.GaussianNB()
    score = norm_classifier.fit(reshape_x, reshape_y).predict_proba(reshape_x)
    estimated = norm_classifier.fit(reshape_x, reshape_y).predict(reshape_x)

    false_pos = dict()
    true_pos = dict()
    class_score = dict()

    for i in range(2):
        false_pos[i], true_pos[i], _ = met.roc_curve(reshape_y - 1, score[:, i])
        class_score[i] = met.roc_auc_score(reshape_y - 1, score[:, i])

    plt.figure()
    plt.plot(false_pos[0], true_pos[0], label="Class 1")
    plt.plot(false_pos[1], true_pos[1], label="Class 2")
    plt.legend()
    plt.show()

    # PART D

    class_flip = lambda x: 0 if x == 2 else 1
    flip_func = np.vectorize(class_flip)
    flip_y = flip_func(reshape_y)
    flip_est = flip_func(estimated)

    emperical_array = np.ndarray((len(flip_y), 2))
    for i in range(1, len(flip_y)):
        if met.confusion_matrix(flip_y[0:i], flip_est[0:i]).shape == (1, 1):
            emperical_array[i, 0] = 0
            emperical_array[i, 1] = 0
        else:
            tn, fp, fn, tp = met.confusion_matrix(flip_y[0:i], flip_est[0:i]).ravel()
            emperical_array[i, 0] = fp / (fp + tn)
            emperical_array[i, 1] = tp / (tp + fn)

    plt.figure()
    plt.plot(emperical_array[:, 0], emperical_array[:, 1])
    plt.show()

    # PART F

    ll_f = sum(np.log((3/4 * sci.norm.pdf(class_1['x'].values, -3, 1)) + (1/4 * sci.norm.pdf(class_1['x'].values, 7, 0.316))))
    ll_a = sum(sci.norm.logpdf(class_1['x'].values, mu_1, var_1))

    print("Class 1 F: %f" % ll_f)
    print("Class 1 A: %f" % ll_a)

    # PART G
    n = len(reshape_y)
    LR = ll_f/ll_a
    BIC_a = math.log(n) * 3 - 2 * math.log(-ll_a)
    BIC_f = math.log(n) * 5 - 2 * math.log(-ll_f)
    BR = BIC_f / BIC_a

    print("Likelihood Ratio: %f" % LR)
    print("BIC A: %f | BIC F: %f" % (BIC_a, BIC_f))
    print("Bayes Factor: %f" % BR)


def six_main(data):
    class_1 = data[data['class'] == 1].reset_index()
    class_2 = data[data['class'] == 2].reset_index()
    class_3 = data[data['class'] == 3].reset_index()

    # PART A

    plt.figure()
    plt.scatter(class_1['x'], class_1['y'], label="Class 1")
    plt.scatter(class_2['x'], class_2['y'], label="Class 2")
    plt.scatter(class_3['x'], class_3['y'], label="Class 3")
    plt.legend()
    plt.show()

    # PART B

    prior_1 = len(class_1['x']) / (len(data['x']))
    prior_2 = len(class_2['x']) / (len(data['x']))
    prior_3 = len(class_3['x']) / (len(data['x']))

    print("Priors: Class 1 - %f | Class 2 - %f | Class 3 - %f" % (prior_1, prior_2, prior_3))

    # PART C-E

    max_x = data['x'].max()
    min_x = data['x'].min()
    max_y = data['y'].max()
    min_y = data['y'].min()

    trans_x = data['x'].transform(lambda x: (x - min_x) / (max_x - min_x))
    trans_y = data['y'].transform(lambda x: (x - min_y) / (max_y - min_y))

    reshape_x = trans_x.values.reshape(-1, 1)
    reshape_y = trans_y.values.reshape(-1, 1)
    reshape_class = data['class'].values.reshape(-1, 1).ravel()

    reshape_data = np.append(reshape_y, reshape_x, axis=1)

    nb_classifier = nb.MultinomialNB()
    nb_fit = nb_classifier.fit(reshape_data, reshape_class)
    nb_score = nb_fit.score(reshape_data, reshape_class)
    nb_est = nb_fit.predict(reshape_data)
    nb_conf = met.confusion_matrix(reshape_class, nb_est)
    print("Naive Bayes - Score: %f" % nb_score)
    print(nb_conf)

    qda_classifier = da.QuadraticDiscriminantAnalysis()
    qda_fit = qda_classifier.fit(reshape_data, reshape_class)
    qda_score = qda_fit.score(reshape_data, reshape_class)
    qda_est = qda_fit.predict(reshape_data)
    qda_conf = met.confusion_matrix(reshape_class, qda_est)
    print("QDA - Score: %f" % qda_score)
    print(qda_conf)

    lda_classifier = da.LinearDiscriminantAnalysis()
    lda_fit = lda_classifier.fit(reshape_data, reshape_class)
    lda_score = lda_fit.score(reshape_data, reshape_class)
    lda_est = lda_fit.predict(reshape_data)
    lda_conf = met.confusion_matrix(reshape_class, lda_est)
    print("LDA - Score: %f" % lda_score)
    print(lda_conf)

    plt.figure()
    mlxplt.plot_decision_regions(reshape_data, reshape_class, clf=nb_fit)

    plt.figure()
    mlxplt.plot_decision_regions(reshape_data, reshape_class, clf=qda_fit)

    plt.figure()
    mlxplt.plot_decision_regions(reshape_data, reshape_class, clf=lda_fit)

    plt.show()


def main():
    five_data = pd.read_csv('data_3_5.csv', names=['x', 'y'])
    six_data = pd.read_csv('data_3_6.csv', names=['x', 'y', 'class'])

    five_main(five_data)
    six_main(six_data)


if __name__ == "__main__":
    main()
