import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import swifter


class geneExtractLukk:

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

		pca1 = np.delete(X,[1,2,3],axis=1)
		pca2 = np.delete(X,[0,2,3],axis=1)
		pca3 = np.delete(X,[0,1,3],axis=1)
		pca3 = np.delete(X,[0,1,2],axis=1)

		print (np.shape(pca1))


		sns.scatterplot(x=pca1.ravel(),y=pca2.ravel(),color='b')
		#sns.scatterplot(x=pca1.ravel(),y=pca3.ravel(),color='g')
		#sns.scatterplot(x=pca1.ravel(),y=pca2.ravel(),color='b',alpha=0.1)
		plt.show()


class geneExtractOwn:

	def __init__(self):

		df = pd.read_csv('~/IS/Data/dataownnew.csv')
		print ("Read Done")

		df_supplement = pd.read_excel('~/IS/Data/st.xls')
		print ("Supplementary Read")
		#print (df_supplement.columns)
		#df_supplement = df_supplement.rename(columns={"groupLabel (192 groups) ": "label"})

		self.df_supplement = pd.DataFrame()
		self.df_supplement['label'] = df_supplement['groupLabel (192 groups) ']
		df_supplement = pd.DataFrame()
		#print (df_supplement)

		

		# df = self.dataframe_transpose(df)

		X = df.values		
		X = self.preprocess_remove_text(X)
		X = np.transpose(X)
		
		X_pca = self.pca_transformer(X)
		
		self.data_set(X_pca)
		
		self.pca_plotter()
		
		gc.collect()
		
		
	def data_set(self,X):

		self.df_supplement['pca1'] = np.delete(X,[1,2,3],axis=1)
		self.df_supplement['pca1'] = self.df_supplement['pca1'].swifter.apply(self.shift_axis)
		self.df_supplement['pca2'] = np.delete(X,[0,2,3],axis=1)
		self.df_supplement['pca3'] = np.delete(X,[0,1,3],axis=1)
		self.df_supplement['pca3'] = self.df_supplement['pca3'].swifter.apply(self.shift_axis)
		self.df_supplement['pca4'] = np.delete(X,[0,1,2],axis=1)
		self.df_supplement['pca4'] = self.df_supplement['pca4'].swifter.apply(self.shift_axis)


	def shift_axis(self,x):

		return (x*-1)
	
	def dataframe_transpose(self,data):

		return(data.transpose())

		
	def preprocess_remove_text(self,X):
		
		x_new = np.delete(X,[0],axis=1)

		return (x_new)


	def pca_transformer(self,X):

		pca = PCA(n_components=4)
		x_new = pca.fit_transform(X)

		return (x_new)


	def pca_plotter(self):	


		#sns.scatterplot(x='pca1',y='pca2',hue='label',data=self.df_supplement,legend=False)
		sns.scatterplot(x='pca3',y='pca4',hue='label',data=self.df_supplement,legend=False)
		#sns.scatterplot(x=pca1.ravel(),y=pca3.ravel(),color='g')
		#sns.scatterplot(x=pca1.ravel(),y=pca2.ravel(),color='b',alpha=0.1)
		#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
		plt.show()



if __name__ == '__main__':


	#x = geneExtractLukk()
	test = geneExtractOwn()

