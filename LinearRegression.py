import os
import logging
import numpy as np
import tensorflow as tf
import matplotlib.font_manager
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class LinearRegression(object):
    """ This class preforms the analysis required for linear regression
    """

    def __init__(self):
        """ initializer
		
        """
        logging.info('Linear Regression Initializer...')

        # load sample data from sklearn
        self.diabetes = datasets.load_diabetes(return_X_y=True)
        self.n_samples, self.n_features = self.diabetes[0].shape
        self.batch_size = 1000


        self.Optimizer = "GradientDescent"	#"Adam"
        self.learning_rate = 0.5       		# Optimizer Learning Rate
        self.epsilon = 1e-8             	# Optimizer Error
        self.loss_func = "MeanSq" 			#"LeastSq"      # Loss function
        self.iterations = 1000           	# Session run iteration
        self.activation_func = "relu6"  	#"relu"   #"sigmoid"
        self.simulation_name = self.Optimizer + "_LR=" + str(self.learning_rate) + "_Ep=" + str(self.epsilon) + \
                               "_" + self.loss_func + "_itr=" + str(self.iterations) + "_batch=" + str(self.batch_size)\
                                + "_func=" + self.activation_func

        # extract x and y
        self.x, self.y = self.diabetes
        self.x = np.asarray(self.x, dtype=np.float)
        self.y = np.asarray(self.y, dtype=np.float)

        # randomly pick training and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y,
                                                                                test_size=0.2, random_state=42)

        # plt.scatter(self.x_train, self.y_train)
        # plt.show()

    def sk_linear_reg(self):
        """ Linear regression using sklearn API
		
        :return:
		
        """
        logging.info('sklearn Linear Regression ...')

        # Create Linear Regression Class
        linear = linear_model.LinearRegression()

        # Train the model using the training sets and check score
        linear.fit(self.x_train, self.y_train)

        # Predict Output
        predicted = linear.predict(self.x_test)

        # Equation coefficient and intercept
        print("sklearn linear regression technique")
        print("Coefficient: \n", linear.coef_)
        print("Intercept: \n %.2f" % linear.intercept_)
        print("Mean square error: %.2f" % mean_squared_error(self.y_test, predicted))
        print("Variance Score: %.2f" % r2_score(self.y_test, predicted))

        # Graph predicted y vs the real y
        fig, ax = plt.subplots(nrows=2)

        ax[0].scatter(self.x_train[:, 0], self.y_train, c='black', lw=1, label='Training Set')
        ax[0].scatter(self.x_test[:, 0], self.y_test, c='brown', lw=1, label='Test Set')
        ax[0].scatter(self.x_test[:, 0], predicted, c='blue', lw=1, label='Prediction')
        ax[0].set_xlabel('y')
        ax[0].set_ylabel('x')
        handles, labels = ax[0].get_legend_handles_labels()
        ax[0].legend(handles, labels, loc='upper center', ncol=3)

        ax[1].scatter(self.y_test, predicted, edgecolors=(0, 0, 0))
        ax[1].plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'k--', lw=4)
        ax[1].set_xlabel('Measured')
        ax[1].set_ylabel('Predicted')
        #ax[1].legend()

        plt.xticks()
        plt.yticks()
        #plt.show()
        fig.savefig(str("./" + datetime.now().strftime("%Y%m%d%H%M") + '_sk'))
        plt.close()

    def tf_linear_reg(self):
        """ Linear Regression using the tensorflow API
		
        :return:
		
        """
        logging.info("Tensorflow Linear Regression ...")

        n_train_samples = self.x_train.shape[0]
        batch_size = self.batch_size

        # pick only one feature to train
        self.x_train = self.x_train[:, 0]
        self.x_test = self.x_test[:, 0]

        # Tensorflow is finicky about the shapes, so resize the training set (n_samples, 1)
        self.x_train = np.reshape(self.x_train, (n_train_samples, 1))
        self.y_train = np.reshape(self.y_train, (n_train_samples, 1))

        self.x_test = np.reshape(self.x_test, (self.x_test.shape[0], 1))
        self.y_test = np.reshape(self.y_test, (self.x_test.shape[0], 1))

        # Define placeholders for input
        x = tf.placeholder(name="x", dtype=tf.float32, shape=(batch_size, 1))
        y = tf.placeholder(name="y", dtype=tf.float32, shape=(batch_size, 1))

        with tf.name_scope(""):
            # Define Variables to be learned
            try:
                with tf.variable_scope("linear-regression"):
                    W = tf.get_variable("weights", (1, 1), dtype=tf.float32, initializer=tf.random_normal_initializer())
                    b = tf.get_variable("bias", (1,), dtype=tf.float32, initializer=tf.constant_initializer())
                    #y_pre = tf.matmul(x, W) + b
                    #y_pre = tf.nn.relu(tf.matmul(x, W) + b)
                    y_pre = tf.nn.relu6(tf.matmul(x, W) + b)

                    #loss = tf.reduce_sum(tf.pow((y - y_pre), 2) / n_train_samples)
                    loss = tf.losses.mean_squared_error(y, y_pre)

            except NotImplementedError:
                logging.exception('TensorFlow Variables Not Defined')

            # Define Optimizer Operation
            #opt_operation = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=self.epsilon).minimize(loss)
            opt_operation = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)

            try:
                # Define the Sever to run
                server = tf.train.Server.create_local_server()

                with tf.Session(server.target) as sess:
                    # Setup summary files
                    tf.summary.scalar("loss", loss)
                    tf.summary.histogram("weights", W)
                    tf.summary.histogram("bias", b)

                    # Merge Summary files
                    merged_summary = tf.summary.merge_all()

                    # Setup visualizing graph and reports
                    writer = tf.summary.FileWriter("./out/" + self.simulation_name)
                    writer.add_graph(sess.graph)

                    # Initialize variables in graph
                    sess.run(tf.global_variables_initializer())

                    # Gradient descent loop for 500 steps
                    for _ in range(self.iterations):
                        # Select random mini-batch
                        indices = np.random.choice(n_train_samples, batch_size)
                        x_batch, y_batch = self.x_train[indices], self.y_train[indices]

                        if _ % 5 == 0:
                            s = sess.run(merged_summary, feed_dict={x: x_batch, y: y_batch})
                            writer.add_summary(s, _)

                        # Do gradient descent step
                        sess.run([opt_operation, loss], feed_dict={x: x_batch, y: y_batch})

                    # Print the Model Coefficient & Intercept
                    print("Tensorflow Linear Regression Technique")
                    print("Coefficient: ", W.eval())
                    print("Intercept: ", b.eval())

                    # Estimate the final output for the training and test sets
                    y_train_pred = np.matmul(self.x_train, W.eval()) + b.eval()
                    y_test_pred = np.matmul(self.x_test, W.eval()) + b.eval()

                # Graph the results
                fig, ax = plt.subplots(nrows=2)

                # Graph the training and test sets with the predicted model
                ax[0].scatter(self.x_train, self.y_train, c='black', lw=1, label='Training Set')
                ax[0].scatter(self.x_test, self.y_test, c='brown', lw=1, label='Test Set')
                ax[0].scatter(self.x_train, y_train_pred, c='blue', lw=1, label='Prediction')
                ax[0].scatter(self.x_test, y_test_pred, c='blue', lw=1)
                ax[0].set_xlabel('y')
                ax[0].set_ylabel('x')
                handles, labels = ax[0].get_legend_handles_labels()
                ax[0].legend(handles, labels, loc='upper center', ncol=3)

                # Compare the predicted model and the measured data
                ax[1].scatter(self.y_test, y_test_pred, edgecolors=(0, 0, 0))
                ax[1].plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'k--', lw=4)
                ax[1].set_xlabel('Measured')
                ax[1].set_ylabel('Predicted')
                #ax[1].legend()

                plt.xticks()
                plt.yticks()
                #plt.show()
                fig.savefig(str("./out/" + self.simulation_name + "/" + datetime.now().strftime("%Y%m%d%H%M") + '_tf'))
                plt.close()
            except NotImplementedError:
                logging.exception('TensorFlow Session Not Operated')
