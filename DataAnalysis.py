import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Initializer(object):
    """ This is a class used for data analysis.
    """

    def __init__(self):
        """This is the constructor method.
        """ 

    def open_file(self, filename):
        """ This function reads a csv file into pandas dataframe

        :param filename: name of the file

        :return: dataframe
        """
        return pd.read_csv(filename)

    def scatterplot(self, x, y):
        """ This function plots each variable x w.r.t to single output y variable
            The results will be saved on the local drive
		
		:param x:
		:param y:
		
        :return: N/A
        """
        for i in range(0, len(x.columns) - 1):
            plt.scatter(y[y.columns[0]], x[x.columns[i]])
            plt.ylabel(x.columns[i])
            plt.xlabel(y.columns[0])
            plt.savefig(str(i), format=None)
            plt.show()
            plt.close()

    def pivot(self, x, y, pivot_indices, pivot_values, pivot_columns, pivot_filters, pivot_action):
        """ This function creates a pivot table of data
		
		:param x: 
		:param y:
		:param pivot_indices: list of row labels
        :param pivot_values: list of values
        :param pivot_columns: list of columns
        :param pivot_filters: list of filters
        :param pivot_action: action

        :return:
        """
        self.table = pd.pivot_table(x, index=pivot_indices, values=pivot_values,
                                    columns=pivot_columns, filters=pivot_filters, aggfunc=pivot_action,
                                    fill_value=0)
        self.table.tail(10)

    def bar_chart(self, x, y):
        """ Graphs a bar chart
            arrays should be tweaked internally

		:param x: 
		:param y:
			
        :return:
        """
        a = y[y.columns[0]]  			# first array
        aerr = y[y.columns[0]] / 10  	# first array - error bar
        b = y[y.columns[0]] / 2  		# second array
        berr = y[y.columns[0]] / 20  	# second array - error bar

        ind = np.arange(len(a))  		# the x locations for the groups
        width = 0.35  					# the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(ind - width / 3, a, yerr=aerr, color='SkyBlue', label='')
        rects2 = ax.bar(ind + width / 3, b, yerr=berr, color='IndianRed', label='')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Scores')
        ax.set_title('Scores by group and gender')
        ax.set_xticks(ind)
        # ax.set_xticklabels() 		# put your label array in here
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