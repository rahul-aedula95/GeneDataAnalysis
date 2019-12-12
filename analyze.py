import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import swifter
from MulticoreTSNE import MulticoreTSNE as TSNE
import multiprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

class geneExtractLukk:

	def __init__(self):

		df = pd.read_csv('~/IS/Data/lukk.csv',sep="\t")
		df_supplement = pd.read_csv('~/IS/Data/AnnoEmtab62_new.csv',sep=",")
		
		print (df_supplement)

		
		df = df.T
		
		df = df.drop([0],axis=1)
		df = df.drop(df.index[0])
		
		self.df_supplement = pd.DataFrame()
		#self.df_supplement['label'] = df_supplement['Factor.Value.4.groups.from.blood.to.incompletely.diff.']
		self.df_supplement['label'] = df_supplement['Characteristics.4.meta.groups.']

		print (df.values)
		
		X = df.values		
		self.df = X
		X_pca = self.pca_transformer(X)
		self.data_set(X_pca)
		#self.pca_plotter()

		gc.collect()
		
	
	def pca_transformer(self,X):

		pca = PCA(n_components=4)
		x_new = pca.fit_transform(X)

		return (x_new)

	def shift_axis(self,x):

		return (x*-1)
	

	def data_set(self,X):

		self.df_supplement['lpca1'] = np.delete(X,[1,2,3],axis=1)
		#self.df_supplement['lpca1'] = self.df_supplement['lpca1'].swifter.apply(self.shift_axis)
		self.df_supplement['lpca2'] = np.delete(X,[0,2,3],axis=1)
		#self.df_supplement['lpca2'] = self.df_supplement['lpca2'].swifter.apply(self.shift_axis)
		self.df_supplement['lpca3'] = np.delete(X,[0,1,3],axis=1)
		#self.df_supplement['lpca3'] = self.df_supplement['lpca3'].swifter.apply(self.shift_axis)
		self.df_supplement['lpca4'] = np.delete(X,[0,1,2],axis=1)
		#self.df_supplement['lpca4'] = self.df_supplement['lpca4'].swifter.apply(self.shift_axis)



	def pca_plotter(self):	


		sns.scatterplot(x='pca1',y='pca2',hue='label',data=self.df_supplement)
		#sns.scatterplot(x='pca3',y='pca4',hue='label',data=self.df_supplement)
		#sns.scatterplot(x=pca1.ravel(),y=pca3.ravel(),color='g')
		#sns.scatterplot(x=pca1.ravel(),y=pca2.ravel(),color='b',alpha=0.1)
		#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
		plt.show()

	def get_frame(self):

		return(self.df)

class geneExtractOwn:

	def __init__(self):

		

		df = pd.read_csv('~/IS/Data/dataownnew.csv')
		print ("Read Done")

		df_supplement = pd.read_excel('~/IS/Data/st.xls')
		print ("Supplementary Read")
		#print (df_supplement.columns)
		

		self.df_supplement = pd.DataFrame()
		self.df_supplement['label'] = df_supplement[['Large-scale group','groupLabel (192 groups) ']].apply(self.liver_sample_initiate,axis=1)

		

		df_supplement = pd.DataFrame()
		
		X = df.values		
		X = self.preprocess_remove_text(X)
		X = np.transpose(X)
		self.X = X
		
		
		
		#self.pca_plotter()
		
		gc.collect()
		
	

	def collect_components(self,components):

		self.components = components

		X_pca = self.pca_transformer(self.X)
		
		df = self.data_set(X_pca)

		df['label'] = self.df_supplement['label']

		return(df)

		

	def data_set(self,X):

		df_supplement = pd.DataFrame()

		delete_list = list(range(0,self.components))

		index = 0
		for i in range(0,self.components):

			col_name = 'pca' + str(i)
			#print (col_name)
			#print (np.shape(np.delete(X,[j for j in delete_list if j !=i],axis=1)))
			k = np.delete(X,[j for j in delete_list if j !=i],axis=1).ravel()
			#print (len(k))
			df_supplement[col_name] = k
			#df_supplement[col_name] = np.delete(X,[j for j in delete_list if j !=i],axis=1).reshape(-1,1)
			index +=1
			#print (df_supplement)

		col_name = 'pca' + str(index)
		k = np.delete(X,[j for j in delete_list if j !=index],axis=1).ravel()
		
		return (df_supplement)
			





		#self.df_supplement['pca1'] = np.delete(X,[1,2,3],axis=1)
		#self.df_supplement['pca1'] = self.df_supplement['pca1'].swifter.apply(self.shift_axis)
		#self.df_supplement['pca2'] = np.delete(X,[0,2,3],axis=1)
		#self.df_supplement['pca3'] = np.delete(X,[0,1,3],axis=1)
		#self.df_supplement['pca3'] = self.df_supplement['pca3'].swifter.apply(self.shift_axis)
		#self.df_supplement['pca4'] = np.delete(X,[0,1,2],axis=1)
		#self.df_supplement['pca4'] = self.df_supplement['pca4'].swifter.apply(self.shift_axis)

	def liver_sample_initiate(self, x):

		if 'Liver' in str(x[1]):

			return 'Liver'
		else:
			return x[0]	


	def shift_axis(self,x):

		return (x*-1)
	
	def dataframe_transpose(self,data):

		return(data.transpose())

		
	def preprocess_remove_text(self,X):
		
		x_new = np.delete(X,[0],axis=1)

		return (x_new)


	def pca_transformer(self,X):

		pca = PCA(n_components=self.components)
		x_new = pca.fit_transform(X)

		return (x_new)


	def pca_plotter(self):	


		sns.scatterplot(x='pca1',y='pca2',hue='label',data=self.df_supplement)
		#sns.scatterplot(x='pca3',y='pca4',hue='label',data=self.df_supplement,alpha=0.9)
		#sns.scatterplot(x=pca1.ravel(),y=pca3.ravel(),color='g')
		#sns.scatterplot(x=pca1.ravel(),y=pca2.ravel(),color='b',alpha=0.1)
		#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
		plt.show()

	def get_frame(self):

		return(self.df_supplement)


class correlateDatasets:

	def __init__(self,df1,df2):

		
		df1 = self.preprocess_frame(df1)
		df2 = self.preprocess_frame(df2)
		frames = [df1,df2]
		self.final_frame = pd.concat(frames, axis=1, ignore_index=False, sort=False)
		print(self.final_frame)
		#print (df1['pca1'].shape)
		#print (df2['lpca1'].shape)
		#print (df1['pca1'].corr(df2['lpca1']))
		self.plot_correlation()
		gc.collect()

	def preprocess_frame(self,df):

		return(df.drop(['label'],axis=1))



	def plot_correlation(self):

		#sns.lineplot(x=self.final_frame.index,y='pca1',data=self.final_frame)
		#sns.lineplot(y='lpca1',x=self.final_frame.index,data=self.final_frame)
		# df3 = pd.DataFrame()

		# x=self.final_frame.index
		# y = list(self.final_frame['pca1'])
		# y.extend(list(self.final_frame['lpca1']))
		# df3['y'] = y

		# y = df3['y']
		# print (y.shape)
		
		sns.jointplot("lpca1", "pca3", data=self.final_frame, kind="reg")
		#sns.jointplot("lpca2", "pca2", data=self.final_frame, kind="reg")
		#sns.jointplot("lpca3", "pca3", data=self.final_frame, kind="reg")
		#sns.jointplot("lpca4", "pca4", data=self.final_frame, kind="reg")
		# #sns.jointplot(x, y, kind="hex", color="#4CB391")
		# y = y.extend(list(self.final_frame['lpca1']))
		# sns.jointplot(x, y, ki =nd="hex", color="#4CB391")

		#sns.pairplot(self.final_frame,x_vars=["pca1","pca2","pca3","pca4"],y_vars=["lpca1","lpca2","lpca3","lpca4"])
		
		# graph = sns.jointplot(x=x, y=y, color='g')

		# graph.x = x
		# graph.y = self.final_frame['lpca1']
		# graph.plot_joint(plt.scatter, marker='x', c='b', s=50)



		plt.show()

class tsneAnalysis:

	def __init__(self,df,label):

		#df = df.drop(['label'],axis=1)
		self.data = df

		self.tsne_df = pd.DataFrame()
		self.tsne_df['label'] = label

		self.perform_analysis()
		gc.collect()


	def perform_analysis(self):
		self.tsne_prepare(self.data)


	def tsne_prepare(self,df):

		X = df

		print (X)

		tsne = TSNE(n_components=2,perplexity=75,learning_rate=10,n_jobs=multiprocessing.cpu_count(),n_iter=5000)

		Y = tsne.fit_transform(X)

		#print (type(Y))
		k = np.delete(Y,1,axis=1).ravel()
		m = np.delete(Y,0,axis=1).ravel()
		#print (np.shape(k))

		self.tsne_df['x'] = k
		self.tsne_df['y'] = m

		self.plot_results(self.tsne_df,"x","y")


	def plot_results(self,df,x,y):

		sns.scatterplot(x=x,y=y,hue='label',data=df)
		plt.show()

	

class extendedAnalysis:

	def __init__(self,data):

		

		X,y = self.preprcessor(data)

		self.X_train,self.X_test,self.y_train,self.y_test = self.kfold(X,y)


		#print (self.X_test)
		#print (len(self.X_test))
				
		gc.collect()

	def collect_results(self):

		log_acc = self.log_project()
		rf_acc = self.random_forest_project()
		svm_acc = self.svm_project()
		
		return (log_acc,rf_acc,svm_acc)


	def preprcessor(self,df):

		y = df['label']
		y = y.astype('category')
		df = df.drop(['label'],axis=1)		
		X = df.values

		y_encode = y.cat.codes
		return(X,y_encode.values)


	def kfold(self,X,y):

		skf = StratifiedKFold(n_splits=5,shuffle=True)

		for train_index, test_index in skf.split(X, y):
			#print("TRAIN:", train_index, "TEST:", test_index)
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]


		return (X_train,X_test,y_train,y_test)


	def confusion(self,y_pred,y_true):
		y_true = pd.Series(y_true)
		y_pred = pd.Series(y_pred)

		print (pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

	def log_project(self):
		
		clf = LogisticRegression(solver='saga',class_weight='balanced', n_jobs=8,penalty='l1')
		clf.fit(self.X_train,self.y_train)
		
		print ("The mean accuracy score for the training data set is ",clf.score(self.X_train,self.y_train))

		y_pred =  clf.predict(self.X_test)


		print ("The mean accuracy score for the testing data set is ",accuracy_score(y_pred,self.y_test))

		# prob =  clf.predict_proba(self.X_test)
		
		self.confusion(y_pred,self.y_test)

		return (accuracy_score(y_pred,self.y_test))


	def random_forest_project(self):
		
		clf = RandomForestClassifier(n_estimators=25,oob_score=True, n_jobs=multiprocessing.cpu_count(),class_weight='balanced',min_samples_split=20,max_features=4, min_samples_leaf=1,max_leaf_nodes=None)


		clf.fit(self.X_train,self.y_train)

		print ("The mean accuracy score for the training data set is ",clf.score(self.X_train,self.y_train))

		
		y_pred = clf.predict(self.X_test)


		print ("The mean accuracy score for the testing data set is ",accuracy_score(y_pred,self.y_test))

		print ("The OOB score is ",clf.oob_score_)
		
		self.confusion(y_pred,self.y_test)
		print (clf.feature_importances_)

		return (accuracy_score(y_pred,self.y_test))

	def svm_project(self):
		clf = svm.SVC(class_weight='balanced',cache_size=100,kernel='rbf',gamma='scale',probability=True)

		clf.fit(self.X_train,self.y_train)

		print ("The mean accuracy score for the training data set is ",clf.score(self.X_train,self.y_train))

		y_pred =  clf.predict(self.X_test)



		print ("The mean accuracy score for the testing data set is ",accuracy_score(y_pred,self.y_test))

		
	
		self.confusion(y_pred,self.y_test)

		return (accuracy_score(y_pred,self.y_test))

class plotLine:

	def __init__(self,l_val,val,x_axis):


		df = pd.DataFrame()

		df['Principal Components'] = x_axis

		df['Test Accuracy'] = val

		df['label'] = l_val

		self.plot_classifiers(df)
		
	def plot_classifiers(self,df):

		sns.set_style("darkgrid")

		sns.lineplot(x="Principal Components", y="Test Accuracy", hue="label", data=df)

		plt.show()




if __name__ == '__main__':


	#lukk = geneExtractLukk()

	#lukk_frame = lukk.get_frame()
	dataOwn = geneExtractOwn()
	acc = []
	label_acc = []
	#rf_acc = []
	#svm_acc = []
	x_axis = []


	for i in range(4,130):
		mat = dataOwn.collect_components(i)

		analyze = extendedAnalysis(mat)

		l_val,r_val,s_val = analyze.collect_results()

		acc.append(l_val)
		label_acc.append('Logistic Regression')
		acc.append(r_val)
		label_acc.append('Random Forests')
		acc.append(s_val)
		label_acc.append('SVM')
		x_axis.append(i)
		x_axis.append(i)
		x_axis.append(i)
	
	figplot = plotLine(label_acc,acc,x_axis)

	#own_mat = dataOwn.get_frame()

	#tsne = tsneAnalysis(own_mat,label)
	#correlate = correlateDatasets(own_frame,lukk_frame)

	#analyze = extendedAnalysis(own_mat)



