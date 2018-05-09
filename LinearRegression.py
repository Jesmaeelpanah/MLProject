import numpy as np
import pandas as pd
from sklearn import linear_model, datasets, model_selection


class LinearRegression(object):
    """ This class preforms the analysis required for linear regression
    """

    def __init__(self):
        """ initializer
        """
        self.diabetes = datasets.load_diabetes(return_X_y=True)
        self.n_samples, self.n_features = self.diabetes[0].shape
        self.x = self.diabetes[0][:, np.newaxis, 2]
        self.y = self.diabetes[1][:, np.newaxis]
        self.dfx = pd.DataFrame(data=self.x, index=None, dtype=float, columns={'x'})
        self.dfy = pd.DataFrame(data=self.y, index=None, dtype=float)
        self.df = pd.concat([self.dfx, self.dfy], axis=1)
        #self.df = self.df.rename(columns={'0': 'x', '0': 'y'})
        del self.dfx, self.dfy, self.x, self.y

        self.train = self.df.sample(frac=0.8, random_state=200)
        self.test = self.df.drop(self.train.index)

        #self.x_train = self.train.loc[]
        #self.y_train = self.train.loc[2]

        print(self.df)


    def LR(self):
        """ Performs Linear regression
        :return:
        """
