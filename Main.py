import logging
import Initializer as DA
import DimensionalityReduction as DR
import LinearRegression as LR
import Spark as SP
from datetime import datetime

_dataAnalysis = False
_dimReduction = True
_linearReg = False
_spark = False

def configure_logging():
    """ Logger initializer
    :return:'%(asctime)s %(message)s', datefmt='%Y/%m/%d %I:%M:%S %p'
    """
    log_filename = str("./" + datetime.now().strftime("%Y%m%d%H%M") + '_logger.log')
    logging.basicConfig(filename=log_filename,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt='%Y/%m/%d %I:%M:%S %p',
                        level=logging.DEBUG)
    logger = logging.getLogger(__name__)

def main():
    configure_logging()
    logging.info('Started')
	
	def __init__(self):
		"""This is the constructor method.
		"""
		self.x = None
		self.y = None

	
    if _dataAnalysis:
        logging.info('Dimensionality Reduction Analysis')
        data = DA.Initializer()
        self.x = data.open_file("Xvar.csv")
        self.y = data.open_file("Yvar.csv")
        # data.scatterplot(self.x, self.y)
        # data.bar_chart(self.x, self.y)
        # data.pivot()

    if _dimReduction:
		self.digits = datasets.load_digits()
        self.x = self.digits.data
        self.y = self.digits.target
	
        reduction = DR.DimensionalityReduction()
        reduction.PCA(self.x, self.y)
        #reduction.LDA(self.x, self.y)
        #reduction.TSNE(self.x, self.y)
        #reduction.PCA_TSNE(self.x, self.y)
		#reduction.TF_PCA(self.x, self.y)

    if _linearReg:
        logging.info('Linear Regression Analysis ...')
        linear_reg = LR.LinearRegression()
        linear_reg.sk_linear_reg()
        linear_reg.tf_linear_reg()

    if _spark:
        logging.info('Spark Analysis ...')
        sp = SP.Spark()
        sp.spark()
    logging.info('Finished')

if __name__ == '__main__':
    main()