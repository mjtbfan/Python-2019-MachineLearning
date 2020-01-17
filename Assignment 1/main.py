# Nicholas Delli Carpini - CS 4342

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def get_n_mle(cleaned_data):
    mu = cleaned_data.mean()
    temp_sig = 0
    for i in cleaned_data:
        temp_sig += (i - mu) ** 2
    var = temp_sig / len(cleaned_data.index)

    print("Normal: mu=", mu, ", var=", var)
    return mu, var


def get_b_mle(cleaned_data):
    alpha, beta, _, _ = stats.beta.fit(cleaned_data, loc=-10, scale=20)

    print("Beta: alpha=", alpha, ", beta=", beta)
    return alpha, beta


def get_l_mle(cleaned_data):
    shape, loc, scale = stats.lognorm.fit(cleaned_data, loc=-10, scale=20)

    print("Log Norm: mu=", np.log(scale), ", var=", shape ** 2)
    return shape, loc, scale


def get_avg_log(cleaned_data, mu, var, alpha, beta, shape, loc, scale):
    norm_pdfs = []
    beta_pdfs = []
    log_pdfs = []
    n = len(cleaned_data)

    for i in cleaned_data:
        norm_pdfs.append(np.log(stats.norm.pdf(i, mu, math.sqrt(var))))
        beta_pdfs.append(np.log(stats.beta.pdf(i, alpha, beta, loc=-10, scale=20)))
        log_pdfs.append(np.log(stats.lognorm.pdf(i, shape, loc=loc, scale=scale)))

    return np.exp(sum(norm_pdfs)/n), np.exp(sum(beta_pdfs)/n), np.exp(sum(log_pdfs)/n)


def create_graphs(data):
    cleaned_data = data["Rating"]
    mu, var = get_n_mle(cleaned_data)
    alpha, beta = get_b_mle(cleaned_data)
    shape, loc, scale = get_l_mle(cleaned_data)
    temp_x = np.linspace(start=-10, stop=10, endpoint=True)

    plt.figure(1)
    five_plot = cleaned_data.plot(kind="hist", by="Rating", bins=5, density=True)
    plt.plot(temp_x, stats.norm.pdf(temp_x, mu, math.sqrt(var)), label="Norm")
    plt.plot(temp_x, stats.beta.pdf(temp_x, alpha, beta, loc=-10, scale=20), label="Beta")
    plt.plot(temp_x, stats.lognorm.pdf(temp_x, shape, loc=loc, scale=scale), label="Log Norm")
    five_plot.set_xlabel("Rating")
    five_plot.set_title("Rating Histogram | Bins=5")
    five_plot.legend()
    plt.show()

    plt.figure(2)
    ten_plot = cleaned_data.plot(kind="hist", by="Rating", bins=10, density=True)
    plt.plot(temp_x, stats.norm.pdf(temp_x, mu, math.sqrt(var)), label="Norm")
    plt.plot(temp_x, stats.beta.pdf(temp_x, alpha, beta, loc=-10, scale=20), label="Beta")
    plt.plot(temp_x, stats.lognorm.pdf(temp_x, shape, loc=loc, scale=scale), label="Log Norm")
    ten_plot.set_xlabel("Rating")
    ten_plot.set_title("Rating Histogram | Bins=10")
    ten_plot.legend()
    plt.show()

    plt.figure(3)
    hund_plot = cleaned_data.plot(kind="hist", by="Rating", bins=100, density=True)
    plt.plot(temp_x, stats.norm.pdf(temp_x, mu, math.sqrt(var)), label="Norm")
    plt.plot(temp_x, stats.beta.pdf(temp_x, alpha, beta, loc=-10, scale=20), label="Beta")
    plt.plot(temp_x, stats.lognorm.pdf(temp_x, shape, loc=loc, scale=scale), label="Log Norm")
    hund_plot.set_xlabel("Rating")
    hund_plot.set_title("Rating Histogram | Bins=100")
    hund_plot.legend()
    plt.show()

    plt.figure(4)
    thous_plot = cleaned_data.plot(kind="hist", by="Rating", bins=1000, density=True)
    plt.plot(temp_x, stats.norm.pdf(temp_x, mu, math.sqrt(var)), label="Norm")
    plt.plot(temp_x, stats.beta.pdf(temp_x, alpha, beta, loc=-10, scale=20), label="Beta")
    plt.plot(temp_x, stats.lognorm.pdf(temp_x, shape, loc=loc, scale=scale), label="Log Norm")
    thous_plot.set_xlabel("Rating")
    thous_plot.set_title("Rating Histogram | Bins=1000")
    thous_plot.legend()
    plt.show()

    log_norm, log_beta, log_log = get_avg_log(cleaned_data, mu, var, alpha, beta, shape, loc, scale)
    print("\nAverage Probabilities")
    print(" Normal: ", log_norm)
    print(" Beta: ", log_beta)
    print(" Log Norm: ", log_log)


def create_training_graphs(data):
    cleaned_data = data["Rating"]
    shuffled_data = []
    temp_x = np.linspace(start=-10, stop=10, endpoint=True)

    for i in range(10):
        shuffled_data.append(cleaned_data.sample(frac=1/10))

    avg_mu, avg_var = 0, 0
    avg_alpha, avg_beta = 0, 0
    avg_shape, avg_loc, avg_scale = 0, 0, 0
    avg_norm, avg_betadist, avg_log = [], [], []

    for i in range(9):
        print("\nTraining Set %i" % i)

        mu, var = get_n_mle(shuffled_data[i])
        alpha, beta = get_b_mle(shuffled_data[i])
        shape, loc, scale = get_l_mle(shuffled_data[i])

        norm_pdf, beta_pdf, log_pdf = get_avg_log(shuffled_data[i], mu, var, alpha, beta, shape, loc, scale)

        print("Average Probabilities:")
        print(" Normal: ", norm_pdf)
        print(" Beta: ", beta_pdf)
        print(" Log Norm: ", log_pdf)

        avg_mu += mu
        avg_var += var
        avg_alpha += alpha
        avg_beta += beta
        avg_scale += scale
        avg_loc += loc
        avg_shape += shape

        avg_norm.append(norm_pdf)
        avg_betadist.append(beta_pdf)
        avg_log.append(log_pdf)

        plt.figure(i)
        plot = shuffled_data[i].plot(kind="hist", by="Rating", bins=10, density=True)
        plt.plot(temp_x, stats.norm.pdf(temp_x, mu, math.sqrt(var)), label="Norm")
        plt.plot(temp_x, stats.beta.pdf(temp_x, alpha, beta, loc=-10, scale=20), label="Beta")
        plt.plot(temp_x, stats.lognorm.pdf(temp_x, shape, loc=loc, scale=scale), label="Log Norm")
        plot.set_xlabel("Rating")
        plot.set_title("Training Histogram %i | Bins=10" % i)
        plot.legend()
        plt.show()

    avg_mu = avg_mu / 9
    avg_var = avg_var / 9
    avg_alpha = avg_alpha / 9
    avg_beta = avg_beta / 9
    avg_shape = avg_shape / 9
    avg_loc = avg_loc / 9
    avg_scale = avg_scale / 9
    avg_norm_val = sum(avg_norm) / 9
    avg_betadist_val = sum(avg_betadist) / 9
    avg_log_val = sum(avg_log) / 9

    print("\nAverage Values")
    print("Normal: mu = %f, var = %f" % (avg_mu, avg_var))
    print("Beta: alpha = %f, beta = %f" % (avg_alpha, avg_beta))
    print("Log Norm: mu = %f, var = %f" % (np.log(avg_scale), avg_shape ** 2))

    print("\nTest Set")
    test_mu, test_var = get_n_mle(shuffled_data[9])
    test_alpha, test_beta = get_b_mle(shuffled_data[9])
    test_shape, test_loc, test_scale = get_l_mle(shuffled_data[9])
    test_norm, test_betadist, test_log = get_avg_log(shuffled_data[9], test_mu, test_var, test_alpha, test_beta, test_shape, test_loc, test_scale)

    error_norm = (abs(avg_norm_val - test_norm)/test_norm) * 100
    error_beta = (abs(avg_betadist_val - test_betadist)/test_betadist) * 100
    error_log = (abs(avg_log_val - test_log)/test_log) * 100

    print("\nTest Probabilities")
    print("Normal:   Avg = %f | Actual = %f" % (avg_norm_val, test_norm))
    print("          Error = %f" % error_norm)
    print("Beta:     Avg = %f | Actual = %f" % (avg_betadist_val, test_betadist))
    print("          Error = %f" % error_beta)
    print("Log Norm: Avg = %f | Actual = %f" % (avg_log_val, test_log))
    print("          Error = %f" % error_log)

    plt.figure(9)
    plot = shuffled_data[9].plot(kind="hist", by="Rating", bins=10, density=True)
    plt.plot(temp_x, stats.norm.pdf(temp_x, avg_mu, math.sqrt(avg_var)), label="Norm")
    plt.plot(temp_x, stats.beta.pdf(temp_x, avg_alpha, avg_beta, loc=-10, scale=20), label="Beta")
    plt.plot(temp_x, stats.lognorm.pdf(temp_x, avg_shape, loc=avg_loc, scale=avg_scale), label="Log Norm")
    plot.set_xlabel("Rating")
    plot.set_title("Test Histogram | Bins=10")
    plot.legend()
    plt.show()

    plt.figure(10)
    fig10, box = plt.subplots()
    box.set_title("Average Probabilities | Medians=Test Set")
    box.boxplot([avg_norm, avg_betadist, avg_log], usermedians=[test_norm, test_betadist, test_log], labels=["Normal", "Beta", "Log Norm"])
    plt.show()


def main():
    # jester_items_clean = open("data/jester_items.clean.dat", "r")
    # jester_items = open("data/jester_items.dat", "r")
    # jester_ratings_jokes = open("data/jester_ratings_jokes.mat", "r")
    # jester_ratings_users = open("data/jester_ratings_users.mat", "r")

    jester_ratings = open("data/jester_ratings.dat", "r")

    ratings = np.loadtxt(jester_ratings)
    ratings_frame = pd.DataFrame(ratings, columns=["User ID", "Joke ID", "Rating"])
    create_graphs(ratings_frame)
    create_training_graphs(ratings_frame)


if __name__ == "__main__":
    main()
