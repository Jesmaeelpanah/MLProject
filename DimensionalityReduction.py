import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets, decomposition, discriminant_analysis


class DimensionalityReduction(object):
    """ This class contains functions that can be used for dimensionality reduction purposes
    """

    def __init__(self):
        self.digits = datasets.load_digits()
        self.x = self.digits.data
        self.y = self.digits.target
        self.n_samples, self.n_feature = self.x.shape
        print(self.n_samples, self.n_feature)

    def embedding_plot(self, x, title):
        """ Plots the reduced array of x with a given title
        :param x: 2D array of modified parameters
        :param title: title of the graph
        :return:
        """
        x_min, x_max = np.min(x, axis=0), np.max(x, axis=0)
        x = (x - x_min) / (x_max - x_min)

        plt.figure()
        ax = plt.subplot(aspect='equal')
        sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=self.y / 10)

        shown_images = np.array([1., 1.])
        '''
        for i in range(x.shape[0]):
            if np.min(np.sum(np.power(x[i] - shown_images, 2), axis=0)) < 1e-2:
                continue
            shown_images = np.r_[shown_images, x[i]]
            ax.add_artist(offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(self.digits.images[i], cmap=plt.cm.gray_r),
                x[i]))
        '''
        plt.xticks([]), plt.yticks([])
        plt.title(title)

    def PCA(self):
        """ Principle Component Analysis Method
        :return:
        """
        x_pca = decomposition.PCA(n_components=2).fit_transform(self.x)
        self.embedding_plot(x_pca, 'PCA')
        plt.show()

    def LDA(self):
        """Linear Discrement Analysis
        :return:
        """
        x_lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2).fit_transform(self.x, self.y)
        self.embedding_plot(x_lda, "LDA")
        plt.show()

    def TSNE(self):
        """ T-distributed neighbour embedding
        :return:
        """
        x_tsne = manifold.TSNE(n_components=2).fit_transform(self.x)
        self.embedding_plot(x_tsne, "TSNE")
        plt.show()

    def PCA_TSNE(self):
        """ Run PCA first and then apply T-distributed neighbour embedding
        :return:
        """
        x_pca = decomposition.PCA(n_components=2).fit_transform(self.x)
        x_tsne = manifold.TSNE(n_components=2).fit_transform(x_pca)
        self.embedding_plot(x_tsne, "PCA_TSNE")
        plt.show()
