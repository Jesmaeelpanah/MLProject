import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class DataAnalysis(object):
    """ This is a class used for data analysis.
    """

    def __init__(self):
        """This is the constructor method.
        """
        self.xvar = None
        self.yvar = None
        self.digits = None
        self.x = None
        self.y = None

    def open_files(self, xvar_filename, yvar_filename):
        """ This function will open the csv files
            x variables (input vars)
            y variable (output var)

        :param xvar_filename: first file name for independant variables
        :param yvar_filename: second file name for dependant variable

        :return:
            xvar -> pandas dataframe (input variables)
            yvar -> pandas dataframe (output variable)
        """
        self.xvar = self.open_file(xvar_filename)
        self.yvar = self.open_file(yvar_filename)

    def open_file(self, filename):
        """ This function opens a csv file into pandas dataframe

        :param filename: name of the file

        :return: dataframe
        """
        return pd.read_csv(filename)

    def plot_x_y_var(self):
        """ This function will plot the each x variable w.r.t to output y variable
            The results will be saved in the drive

        :return: N/A
        """
        for i in range(0, len(self.xvar.columns) - 1):
            plt.scatter(self.yvar[self.yvar.columns[0]], self.xvar[self.xvar.columns[i]])
            plt.ylabel(self.xvar.columns[i])
            plt.xlabel(self.yvar.columns[0])
            plt.savefig(str(i), format=None)
            plt.show()
            plt.close()

    def pivot(self, pivot_indices, pivot_values, pivot_columns, pivot_filters, pivot_action):
        """ This function will create a pivot table of data

        :param pivot_indices: list of row labels
        :param pivot_values: list of values
        :param pivot_columns: list of columns
        :param pivot_filters: list of filters
        :param pivot_action: action

        :return:
        """
        self.table = pd.pivot_table(self.xvar, index=pivot_indices, values=pivot_values,
                                    columns=pivot_columns, filters=pivot_filters, aggfunc=pivot_action,
                                    fill_value=0)
        self.table.tail(10)

    def bar_chart(self):
        """ Graphs a bar charf
            arrays should be tweaked internally

        :return:
        """
        a = self.yvar[self.yvar.columns[0]]  # first array
        aerr = self.yvar[self.yvar.columns[0]] / 10  # first array - error bar
        b = self.yvar[self.yvar.columns[0]] / 2  # second array
        berr = self.yvar[self.yvar.columns[0]] / 20  # second array - error bar

        ind = np.arange(len(a))  # the x locations for the groups
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(ind - width / 3, a, yerr=aerr, color='SkyBlue', label='')
        rects2 = ax.bar(ind + width / 3, b, yerr=berr, color='IndianRed', label='')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Scores')
        ax.set_title('Scores by group and gender')
        ax.set_xticks(ind)
        # ax.set_xticklabels() # put your label array in here
        ax.legend()

        def autolabel(rects, xpos='center'):
            """ Attach a text label above each bar in *rects*, displaying its height.
            *xpos* indicates which side to place the text w.r.t. the center of
            the bar. It can be one of the following {'center', 'right', 'left'}.

            :param rects:
            :param xpos:

            :return:
            """
            xpos = xpos.lower()  # normalize the case of the parameter
            ha = {'center': 'center', 'right': 'left', 'left': 'right'}
            offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() * offset[xpos], 1.01 * height,
                        '{}'.format(height), ha=ha[xpos], va='bottom')

        autolabel(rects1, "left")
        autolabel(rects2, "right")
        plt.show()