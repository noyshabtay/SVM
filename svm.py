#################################
# Your name: Noy Shabtay
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

"""
Q4.1 skeleton.

Please use the provided functions signature for the SVM implementation.
Feel free to add functions and other code, and submit this file with the name svm.py
"""

# generate points in 2D
# return training_data, training_labels, validation_data, validation_labels
def get_points():
    X, y = make_blobs(n_samples=120, centers=2, random_state=0, cluster_std=0.88)
    return X[:80], y[:80], X[80:], y[80:]


def create_plot(X, y, clf):
    plt.clf()

    # plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.PiYG)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0] - 2, xlim[1] + 2, 30)
    yy = np.linspace(ylim[0] - 2, ylim[1] + 2, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])


def train_three_kernels(X_train, y_train, X_val, y_val):
    """
    Returns: np.ndarray of shape (3,2) :
                A two dimensional array of size 3 that contains the number of support vectors for each class(2) in the three kernels.
    """
    C = 1000 # for all models penalty constant is set to 1000.
    kernels = ['linear', 'poly', 'rbf']
    SVs = np.zeros((3, 2))
    for i in range(len(kernels)):
        svc = svm.SVC(kernel=kernels[i], C=C, degree=2).fit(X_train, y_train)
        vectors = svc.predict(svc.support_vectors_).tolist()
        SVs[i, 0], SVs[i, 1] = vectors.count(0), vectors.count(1)
        create_plot(X_train, y_train, svc)
        plt.title('{} Kernel'.format(kernels[i]).capitalize())
        plt.savefig('a_{}.png'.format(kernels[i]))
    return SVs


def linear_accuracy_per_C(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    accuracy = []
    Cs = [np.math.pow(10, factor) for factor in np.arange(-5, 6)]
    for C in Cs:
        svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
        svc.predict(X_val)
        create_plot(X_train, y_train, svc)
        plt.title('C = {}'.format(C))
        plt.savefig('b_{}.png'.format(C))
        accuracy.append(np.sum(svc.predict(X_val) == y_val) / np.size(X_val, 0))
    plt.clf()
    plt.plot(Cs, accuracy)
    plt.ylabel('Accuracy')
    plt.xlabel('C')
    plt.xscale('log')
    plt.savefig('b.png')
    return accuracy


def rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    accuracy = []
    gammas = [np.math.pow(10, factor) for factor in np.arange(-5, 6)]
    for gamma in gammas:
        svc = svm.SVC(kernel='rbf', C=10, gamma=gamma).fit(X_train, y_train)
        svc.predict(X_val)
        create_plot(X_train, y_train, svc)
        plt.title('Gamma = {}'.format(gamma))
        plt.savefig('c_{}.png'.format(gamma))
        accuracy.append(np.sum(svc.predict(X_val) == y_val) / np.size(X_val, 0))
    plt.clf()
    plt.plot(gammas, accuracy)
    plt.ylabel('Accuracy')
    plt.xlabel('Gamma')
    plt.xscale('log')
    plt.savefig('c.png')
    return accuracy

if __name__ == "__main__":
    data = get_points()
    print('Section a')
    print(train_three_kernels(data[0], data[1], data[2], data[3]))
    print('Section b')
    print(linear_accuracy_per_C(data[0], data[1], data[2], data[3]))
    print('Section c')
    print(rbf_accuracy_per_gamma(data[0], data[1], data[2], data[3]))