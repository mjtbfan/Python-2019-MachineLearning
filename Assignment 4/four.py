import numpy as np
import pandas as pd
import sklearn.naive_bayes as nb
import sklearn.discriminant_analysis as da
import sklearn.model_selection as ms
import sklearn.metrics as met
import matplotlib.pyplot as plt
import mlxtend.plotting as mlxplt


def main():
    data = pd.read_csv('data_3_6.csv', names=['x', 'y', 'class'])

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
    nb_scores = ms.cross_val_score(nb_fit, reshape_data, reshape_class, cv=10)
    nb_est = ms.cross_val_predict(nb_fit, reshape_data, reshape_class, cv=10)
    nb_conf = met.confusion_matrix(reshape_class, nb_est)
    print("Naive Bayes - Score %f +/-%f" % (np.mean(nb_scores), np.std(nb_scores)))
    print(nb_conf, "\n")

    qda_classifier = da.QuadraticDiscriminantAnalysis()
    qda_fit = qda_classifier.fit(reshape_data, reshape_class)
    qda_scores = ms.cross_val_score(qda_fit, reshape_data, reshape_class, cv=10)
    qda_est = ms.cross_val_predict(qda_fit, reshape_data, reshape_class, cv=10)
    qda_conf = met.confusion_matrix(reshape_class, qda_est)
    print("QDA - Score %f +/-%f" % (np.mean(qda_scores), np.std(qda_scores)))
    print(qda_conf, "\n")

    lda_classifier = da.LinearDiscriminantAnalysis()
    lda_fit = lda_classifier.fit(reshape_data, reshape_class)
    lda_scores = ms.cross_val_score(lda_fit, reshape_data, reshape_class, cv=10)
    lda_est = ms.cross_val_predict(lda_fit, reshape_data, reshape_class, cv=10)
    lda_conf = met.confusion_matrix(reshape_class, lda_est)
    print("LDA - Score %f +/-%f" % (np.mean(lda_scores), np.std(lda_scores)))
    print(lda_conf, "\n")

    plt.figure()
    mlxplt.plot_decision_regions(reshape_data, reshape_class, clf=nb_fit)

    plt.figure()
    mlxplt.plot_decision_regions(reshape_data, reshape_class, clf=qda_fit)

    plt.figure()
    mlxplt.plot_decision_regions(reshape_data, reshape_class, clf=lda_fit)

    plt.show()


if __name__ == "__main__":
    main()