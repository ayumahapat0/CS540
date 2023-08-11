import copy
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.cluster.hierarchy import dendrogram, linkage


# Takes in a string with a path to a CSV file, and returns the
# data points as a list of dicts
def load_data(filepath):
    res = []
    with open(filepath) as file:
        csvreader = csv.reader(file)
        keys = []

        for row in csvreader:
            dict = {}
            if row[0] == '#':
                keys = copy.deepcopy(row)
                continue

            for i in range(len(keys)):
                dict[keys[i]] = row[i]
            res.append(dict)
    return res


# Takes in one row dict from the data loaded from the previous
# function then calculates the corresponding feature vector for that Pokemon as
# specified above, and returns it as a numpy array of shape (6,). The dtype of this
# array should be int64
def calc_features(row):
    list = []
    list.append(int(row["Attack"]))
    list.append(int(row["Sp. Atk"]))
    list.append(int(row["Speed"]))
    list.append(int(row["Defense"]))
    list.append(int(row["Sp. Def"]))
    list.append(int(row["HP"]))

    return np.array(list, dtype=np.int64)


# Performs complete linkage hierarchical agglomerative clustering
# on the Pokemon with the (x1, . . . , x6) feature representation, and returns a numpy
# array representing the clustering
def hac(features):
    clusters = {}  # dict which has cluster indexes as keys and the number of pokemon in the cluster as values
    cnt = 0
    for n in range(len(features)):
        clusters[cnt] = 1
        cnt += 1

    output = [[0] * 4 for i in range(len(features) - 1)]

    # create distance matrix
    ones = np.ones((len(features), len(features[0])))
    g = np.square(np.asarray(features))
    comp1 = np.dot(g, np.transpose(ones))
    comp2 = np.dot(ones, np.transpose(g))
    comp3 = -2 * np.dot(features, np.transpose(features))
    distance = np.sqrt(comp1 + comp2 + comp3)
    for i in range(len(distance)):
        distance[i][i] = np.Infinity

    # pad distance matrix to make room for clustered adjacency
    new_shape = (len(distance) * 2, len(distance) * 2)
    shape_diff = np.array(new_shape) - np.array(np.asmatrix(distance).shape)
    newrowcol = len(distance)
    distance = np.lib.pad(distance, ((0, shape_diff[0]), (0, shape_diff[1])), 'constant', constant_values=np.Infinity)
    visited = set()

    for i in range(len(features) - 1):
        r = i
        # find minimum distance
        # minIndex = np.unravel_index(np.argmin(np.asmatrix(distance), axis=None), np.asmatrix(distance).shape)
        minIndex = []
        mini = np.Infinity
        for row in range(len(features) + i):
            for col in range(row + 1, len(features) + i):
                if row not in visited and col not in visited and mini > distance[row][col]:
                    mini = distance[row][col]
                    minIndex = (row, col)

        visited.add(minIndex[0])
        visited.add(minIndex[1])

        # update list

        output[r][0] = min(minIndex)
        output[r][1] = max(minIndex)
        output[r][2] = distance[minIndex[0]][minIndex[1]]
        output[r][3] = clusters[output[i][0]] + clusters[output[i][1]]

        clusters[newrowcol] = output[i][3]

        # remove distances
        distance[minIndex[0]][minIndex[1]] = np.Infinity
        distance[minIndex[1]][minIndex[0]] = np.Infinity

        # add new distances

        for k in range(newrowcol):
            distance[newrowcol][k] = distance[k][newrowcol] = max(distance[minIndex[0]][k], distance[minIndex[1]][k])

        for n in range(len(features)):
            distance[minIndex[0]][n] = distance[n][minIndex[0]] = np.Infinity
            distance[minIndex[1]][n] = distance[n][minIndex[1]] = np.Infinity
        newrowcol += 1
        # for d in distance:
        #     print(d)
        # temp = [[0] * len(distance)]
        # for i in range(len(distance)):
        #     temp[0][i] = (max(distance[minIndex[0]][i], distance[minIndex[1]][i]))
        # distance = np.append(distance, temp, axis=0)
        # distance = np.append(distance, temp, axis=1)

    return np.array(output)


# Visualizes the hierarchical agglomerative clustering on
# the Pokemon’s feature representation
def imshow_hac(Z, names):
    dendrogram(Z, labels=names, leaf_rotation=90)
    plt.subplot()
    plt.tight_layout()
    plt.show()
    return


if __name__ == '__main__':
    # print(load_data("Pokemon.csv"))
    # dicts = load_data("Pokemon.csv")

    # print(dicts[len(dicts) - 1])
    # print(len(dicts))
    # test_calcFeatures = calc_features(dicts[0])
    # print(test_calcFeatures)
    # print(type(test_calcFeatures))
    features_and_names = [(calc_features(row), row["Name"]) for row in load_data("Pokemon.csv")[:50]]
    Z = hac([row[0] for row in features_and_names])
    names = [row[1] for row in features_and_names]

    imshow_hac(Z, names)