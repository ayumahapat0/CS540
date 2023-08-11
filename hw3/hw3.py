from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    x = np.load(filename)
    x = x - np.mean(x, axis=0)
    return x


def get_covariance(dataset):
    x = np.dot(np.transpose(dataset), dataset)
    x /= (len(dataset) - 1)
    return x


def get_eig(S, m):
    eigenvalue, eigenvector = eigh(S, subset_by_index=[len(S) - m, len(S) - 1])
    eigenvalue = np.sort(eigenvalue)[::-1]
    identity_matrix = np.identity(m, dtype='float')
    Lambda = eigenvalue * identity_matrix
    U = np.flip(eigenvector, 1)
    return Lambda, U


def get_eig_prop(S, prop):
    eigenvalue, eigenvector = eigh(S, subset_by_value=[prop * np.trace(S), np.inf])
    eigenvalue = np.sort(eigenvalue)[::-1]
    identity_matrix = np.identity(len(eigenvalue), dtype='float')
    Lambda = eigenvalue * identity_matrix
    U = np.flip(eigenvector, 1)
    return Lambda, U


def project_image(image, U):
    projection = np.dot(np.transpose(U), image)
    pca = np.dot(U, projection)
    return pca


def display_image(orig, proj):
    orig = np.reshape(orig, (32, 32))
    orig = np.transpose(orig)
    proj = np.reshape(proj, (32, 32))
    proj = np.transpose(proj)

    figure, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Original')
    ax2.set_title('Projection')

    vax1 = ax1.imshow(orig, aspect='equal')
    vax2 = ax2.imshow(proj, aspect='equal')

    figure.colorbar(vax1, ax=ax1)
    figure.colorbar(vax2, ax=ax2)

    plt.show()
