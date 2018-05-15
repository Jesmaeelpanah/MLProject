import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn
from sklearn import linear_model, datasets, model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class LinearRegression(object):
    """ This class preforms the analysis required for linear regression
    """

    def __init__(self):
        """ initializer
        """
        # load sample data from sklearn
        self.diabetes = datasets.load_diabetes(return_X_y=True)
        self.n_samples, self.n_features = self.diabetes[0].shape

        # extract x and y
        self.x, self.y = self.diabetes
        self.x = np.asarray(self.x, dtype=np.float)
        self.y = np.asarray(self.y, dtype=np.float)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)

        #print(self.x.shape)

    def linear_reg(self):
        """ Performs Linear regression
        :return:
        """

        #self.x_train = self.x_train.reshape((self.x_train.shape[1], self.x_train.shape[0]))
        #self.y_train = self.y_train.reshape((1, self.y_train.shape[0]))

        print(self.y_train.shape)
        #plt.scatter(self.x_train, self.y_train)
        #plt.show()

        # Create Linear Regression Class
        linear = linear_model.LinearRegression()

        # Train the model using the training sets and check score
        linear.fit(self.x_train, self.y_train)

        # Predict Output
        predicted = linear.predict(self.x_test)

        # Equation coefficient and intercept
        print("Coefficient: \n", linear.coef_)
        print("Intercept: \n %.2f" % linear.intercept_)
        print("Mean square error: %.2f" % mean_squared_error(self.y_test, predicted))
        print("Variance Score: %.2f" % r2_score(self.y_test, predicted))

        # Graph predicted y vs the real y
        fig, ax = plt.subplots(nrows=2)

        ax[0].scatter(self.x_test[:, 2], self.y_test, color='black')
        ax[0].scatter(self.x_test[:, 2], predicted, color='blue', linewidth=3)

        ax[1].scatter(self.y_test, predicted, edgecolors=(0, 0, 0))
        ax[1].plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'k--', lw=4)
        ax[1].set_xlabel('Measured')
        ax[1].set_ylabel('Predicted')

        plt.xticks()
        plt.yticks()
        plt.show()

    def tf_linear_reg(self):
        """
        :return:
        """
        x = tf.placeholder(tf.float32)

        tf.InteractiveSession()
        tf.logging.set_verbosity(tf.logging.ERROR)
        a = tf.random_normal((2, 2))
        b = tf.ones((2, 2))

        #tf.reduce_sum(b, reduction_indices=1).eval()
        #tf.reshape(a, (1, 4)).eval()

        c = tf.matmul(a, b)
        d = tf.constant(5.0) * tf.constant(6.0)

        #print(c.eval())

        with tf.Session() as sess:
            print(sess.run(c))
