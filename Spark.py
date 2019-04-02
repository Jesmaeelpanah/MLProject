import os
import logging
import numpy as np
import pyspark as sp
from datetime import datetime


class Spark(object):
    """
    """

    def __init__(self):
        """ Spark initializer
        """

    def spark(self):
        """
        :return:
        """
        data = [1, 2, 3, 4]
        rDD = sp.parallelize(data, 4)
        #print(rDD)