# MLProject
Repository for ML Project
This project includes implementation of the meachine learning algorithms.

# Main.py
Set up the logging library


# Initializer
	Includes the following functions:

		Read_file(filename): 
			This function reads a csv file into pandas dataframe
		
		scatterplot(x, y):
			This function plots each variable x w.r.t to single output y variable
			The results will be saved on the local drive
			
		pivot(x, y, pivot_indices, pivot_values, pivot_columns, pivot_filters, pivot_action):
			This function creates a pivot table of data
			
		bar_chart(x, y):
			Graphs a bar chart (Might need some manipulation depending on the data)
			

# DimensionalityReduction:
	Includes the following functions:
		
		embedding_plot(x, y, title):
			Internal Plotter
				
		PCA(x, y):
			Principle Component Analysis Method
			
		LDA(x, y):
			Linear Discrement Analysis
			
		TSNE(x, y):
			T-distributed neighbour embedding
			
		PCA_TSNE(x, y):
			Run PCA first and then apply T-distributed neighbour embedding
		
		TF_PCA(x, y):
			Using Tensorflow for the PCA analysis

# LinearRegression
	Includes the following functions:
		
		sk_linear_reg():
			Linear regression using sklearn API
		
		tf_linear_reg():
			Linear Regression using the tensorflow API

# Spark
	Includes the following functions:
	
		spark()
			
			
		
		
		
			