import logging
import numpy as np
import tensorflow as tf
import tensorboard as tb
import matplotlib.pyplot as plt
from sklearn import manifold, datasets, decomposition, discriminant_analysis


class DimensionalityReduction(object):
    """ This class contains functions that can be used for dimensionality reduction purposes
    """

    def __init__(self):
		"""This is the constructor method.
        """ 

    def embedding_plot(self, x, y, title):
		""" Normalize and Plotting of variables
		
		:param x:
		:param title:
		
		:return:
		"""
		
		logging.info("Normalize and Plotting of variables")
		
        x_min, x_max = np.min(x, axis=0), np.max(x, axis=0)
        x = (x - x_min) / (x_max - x_min)

        plt.figure()
        ax = plt.subplot(aspect='equal')
        sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=y / 10)

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

    def PCA(self, x, y):
        """ Principle Component Analysis Method

		:param x:
		
        :return:
        """
		
		logging.info("Principle Component Analysis Method")
		
        x_pca = decomposition.PCA(n_components=2).fit_transform(x)
        self.embedding_plot(x_pca, y, 'PCA')
        plt.show()

    def LDA(self, x, y):
        """Linear Discrement Analysis

		:param x:
		
        :return:
        """
		
		logging.info("Linear Discrement Analysis")
		
        x_lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2).fit_transform(x, y)
        self.embedding_plot(x_lda, y, "LDA")
        plt.show()

    def TSNE(self, x, y):
        """ T-distributed neighbour embedding

		:param x:
		
        :return:
        """
		
		logging.info("T-distributed neighbour embedding")
		
        x_tsne = manifold.TSNE(n_components=2).fit_transform(x)
        self.embedding_plot(x_tsne, y, "TSNE")
        plt.show()

    def PCA_TSNE(self, x, y):
        """ Run PCA first and then apply T-distributed neighbour embedding

		:param x:
		
        :return:
        """
		logging.info("Run PCA first and then apply T-distributed neighbour embedding")
		
        x_pca = decomposition.PCA(n_components=2).fit_transform(x)
        x_tsne = manifold.TSNE(n_components=2).fit_transform(x_pca)
        self.embedding_plot(x_tsne, y, "PCA_TSNE")
        plt.show()

    def TF_PCA(self, x, y):
        """ Using Tensorflow for the PCA analysis

		:param x:
		
        :return:
        """
        logging.info("Tensorflow Principal Component Analysis")
		
		n_samples, n_feature = x.shape
        batch_size = 100
		
        tf.svd(name='PCA', full_matrices=x)

        # Define variables to be learned
        # Define placeholders for inputs
        x = tf.placeholder(name='x', dtype=tf.float32, shape=(batch_size, n_feature))
        y = tf.placeholder(name='y', dtype=tf.float32, shape=(batch_size, n_feature))

        try:
            with tf.variable_scope("PCA"):
                w = tf.get_variable((1, 1), name="weights",dtype=tf.float32, initializer=tf.random_normal_initializer())
        except:
            logging.exception('Tensorflow PCA Malfunction')