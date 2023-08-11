import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage


# create dataset from csv file
def load_data(filepath):
    pokemons = []
    with open(filepath, mode="r") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            pokemons.append({key: value for key, value in row.items()})

    return pokemons


# created a array of the specific stats for a pokemon
def calc_features(row):
    stats = []
    stats.append(int(row["Attack"]))
    stats.append(int(row["Sp. Atk"]))
    stats.append(int(row["Speed"]))
    stats.append(int(row["Defense"]))
    stats.append(int(row["Sp. Def"]))
    stats.append(int(row["HP"]))
    pokemon_stats = np.array(stats, dtype=np.int64)
    return pokemon_stats


# Does the clustering
def hac(features):
    # dictionary that keeps track of clusters
    # key: cluster index, value: number of pokemon in cluster
    clusters = {}
    cluster_index = 0
    for n in range(len(features)):
        clusters[cluster_index] = 1
        cluster_index += 1

    # create the distance matrix
    ones_vectors = np.ones((len(features), len(features[0])))
    matrix_prod = np.square(np.asarray(features))
    comp1 = np.dot(matrix_prod, np.transpose(ones_vectors))
    comp2 = np.dot(ones_vectors, np.transpose(matrix_prod))
    comp3 = -2 * np.dot(features, np.transpose(features))
    distance = np.sqrt(comp1 + comp2 + comp3)

    # set all the distance from a cluster to itself to infinity
    for i in range(len(distance)):
        distance[i][i] = np.Infinity

    # create extra padding to distance matrix when new clusters are created
    # to add new distances
    new_shape = (len(distance) * 2, len(distance) * 2)
    shape_difference = np.array(new_shape) - np.array(np.asmatrix(distance).shape)
    # index of the new row added to distance matrix after clustering
    new_row_col_added = len(distance)
    distance = np.lib.pad(distance, ((0, shape_difference[0]), (0, shape_difference[1])), 'constant', constant_values=np.Infinity)

    # keep track of the clusters visited
    clusters_visited = set()
    # create results list
    result = [[0] * 4 for i in range(len(features) - 1)]

    for j in range(len(features) - 1):
        # find the minimum distance
        min_index = []
        minimum = np.Infinity
        for row in range(len(features) + j):
            for col in range(row + 1, len(features) + j):
                if row not in clusters_visited and col not in clusters_visited and minimum > distance[row][col]:
                    minimum = distance[row][col]
                    min_index = (row, col)

        # add clusters visited to the set
        clusters_visited.add(min_index[0])
        clusters_visited.add(min_index[1])

        # update results list
        result[j][0] = min(min_index)
        result[j][1] = max(min_index)
        result[j][2] = distance[min_index[0]][min_index[1]]
        result[j][3] = clusters[result[j][0]] + clusters[result[j][1]]

        clusters[new_row_col_added] = result[j][3]

        # remove distance so we don't grab same locations again in distance matrix
        distance[min_index[0]][min_index[1]] = np.Infinity
        distance[min_index[1]][min_index[0]] = np.Infinity

        # add new distance to distance matrix
        for rowcol in range(new_row_col_added):
            distance[rowcol][new_row_col_added] = distance[new_row_col_added][rowcol] = max(distance[min_index[0]][rowcol], distance[min_index[1]][rowcol])

        for n in range(len(features)):
            distance[min_index[0]][n] = distance[n][min_index[0]] = np.Infinity
            distance[min_index[1]][n] = distance[n][min_index[1]] = np.Infinity

        new_row_col_added += 1

    return np.array(result)


# created the plot showing the process of the clustering
def imshow_hac(Z, names):
    # creates plot
    dendrogram(Z, labels=names, leaf_rotation=90)
    plt.subplot()
    plt.tight_layout()
    plt.show()
    return

