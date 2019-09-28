import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns



class geneExtract:

	def __init__(self):

		df = pd.read_csv('~/Independent_study/Data/datOwnDatasetTranspose.csv')

		X = df.values
		
		X = self.preprocess_remove_text(X)

		X_pca = self.pca_transformer(X)





		
	def preprocess_remove_text(self,X):
		
		x_new = np.delete(X,[0],axis=1)

		return (x_new)


	def pca_transformer(self,X):

		pca = PCA(n_components=2)
		x_new = pca.fit_transform()

		return (x_new)


	def pca_plotter(self,X):







if __name__ == '__main__':


	x = geneExtract()


