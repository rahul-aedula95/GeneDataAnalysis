import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns



class geneExtract:

	def __init__(self):

		df = pd.read_csv('~/IS/Data/lukk.csv',sep="\t")
		df = df.drop([0],axis=0)
		df = df.set_index(['Hybridization REF']) 
		row,col = df.shape
		
		df = df.astype(np.float64)
		print (df.dtypes)

		
		X = df.values		
		X = self.preprocess_remove_text(X)
		X = np.transpose(X)
		X_pca = self.pca_transformer(X)
		self.pca_plotter(X_pca)


		
	def preprocess_remove_text(self,X):
		
		x_new = np.delete(X,[0],axis=1)

		return (x_new)


	def pca_transformer(self,X):

		pca = PCA(n_components=3)
		x_new = pca.fit_transform(X)

		return (x_new)


	def pca_plotter(self,X):

		pca1 = np.delete(X,[1,2],axis=1)
		pca2 = np.delete(X,[0,2],axis=1)
		pca3 = np.delete(X,[0,1],axis=1)

		print (np.shape(pca1))


		sns.scatterplot(x=pca1.ravel(),y=pca2.ravel(),color='b')
		#sns.scatterplot(x=pca1.ravel(),y=pca3.ravel(),color='g')
		#sns.scatterplot(x=pca1.ravel(),y=pca2.ravel(),color='b',alpha=0.1)
		plt.show()




if __name__ == '__main__':


	x = geneExtract()


