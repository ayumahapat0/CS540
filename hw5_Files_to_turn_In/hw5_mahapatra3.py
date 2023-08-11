import csv
import matplotlib.pyplot as plt
import numpy as np


def load_data(filename):
    dataset = []
    with open(filename) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            dataset.append([{key: int(value) for key, value in row.items()}])
    return dataset


def plot_data(data):
    years = []
    days = []
    for point in data:
        years.append(point[0]["year"])
        days.append(point[0]["days"])

    plt.xlabel("Year")
    plt.ylabel("Days")
    plt.plot(years, days)
    plt.savefig("plot.jpg")


def linear_regression(data):
    X = []
    Y = []
    for point in data:
        X.append([1, point[0]["year"]])
        Y.append(point[0]["days"])

    X = np.array(X)
    Y = np.array(Y)

    print("Q3a:")
    print(X)

    print("Q3b:")
    print(Y)

    Z = np.dot(np.transpose(X), X)

    print("Q3c:")
    print(Z)

    inverse = np.linalg.inv(Z)

    print("Q3d:")
    print(inverse)

    PI = np.dot(inverse, np.transpose(X))

    print("Q3e:")
    print(PI)

    B = np.dot(PI, Y)

    print("Q3f:")
    print(B)

    return B


def prediction(lin_reg):
    x_test = 2022
    y_test = lin_reg[0] + lin_reg[1]*x_test

    print("Q4:" + str(y_test))


def model_interpretation(lin_reg):
    if lin_reg[1] < 0:
        print("Q5a: <")
        print("Q5B: the number of ice days per year is decreasing over time")
    elif lin_reg[1] > 0:
        print("Q5a: >")
        print("Q5B: the number of ice days per year is increasing over time")
    else:
        print("Q5a: =")
        print("Q5B: the number of ice days per year is constant over time")


def model_limitation(lin_reg):
    year = (-lin_reg[0] / lin_reg[1])
    print("Q6a: " + str(year))
    print("Q6b: This year makes sense given the trends within the data. From the plot, The number of ice days per year is decreasing, which we also see in our linear regression.")


if __name__ == "__main__":
    datatest = load_data("hw5.csv")
    plot_data(datatest)
    B_test = linear_regression(datatest)
    prediction(B_test)
    model_interpretation(B_test)
    model_limitation(B_test)



