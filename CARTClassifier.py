'''CART实现分类器'''
import numpy as np
import collections

from C45Classifier import C45Classifier
class CARTClassifier(C45Classifier):
	'''CART决策树分类器，与C.5决策树实现的区别在于：
	1._calScore方法中不使用信息增益而是基尼指数
	2.save_model与load_model方法的默认路径
	'''
	def _calScore(self,xtrain,ytrain,feat,splitVal):
		'''改写父类的同名方法,这里采用基尼系数
		先根据特征feat和分割点spliVal将数据集一分为二，然后计算分割前后的指标增益'''	
		xtrain = self._assert_xdata(xtrain)
		ytrain = self._assert_ydata(ytrain)	

		left_data,right_data = self._splitDataSet(xtrain,ytrain,feat,splitVal)
		nexample = float(len(ytrain))
		p_left = len(left_data[1]) / nexample
		p_right = len(right_data[1]) / nexample
		from math import log

		#计算数据集划分前后的基尼指数增益
		baseGini = self._calGini(ytrain)
		newGini = 0
		newGini += self._calGini(left_data[1]) * p_left
		newGini += self._calGini(right_data[1]) * p_right
		giniGain = baseGini - newGini

		return giniGain 

	def save_model(self,path=None):
		'''决策树分类器序列化'''
		if self.bool_not_trained():
			raise self.NotTrainedError('无法进行模型序列化，因为决策树分类器尚未训练!')
		if path is None:
			cur_path = self._reader._dataDir + '/CARTClassifier.pkl'
		else:
			cur_path = path
		import pickle
		with open(cur_path,'wb') as f:
			pickle.dump(self._cur_model,f)
		print('save_model done!')
	
	def load_model(self,path=None):
		'''载入模型'''
		if path is None:
			cur_path = self._reader._dataDir + '/CARTClassifier.pkl'
		else:
			cur_path = path
		import pickle
		with open(cur_path,'rb') as f:
			self._stored_model = pickle.load(f)
		print('load_model done!')

if __name__ == '__main__':
	obj = CARTClassifier(dataDir='/home/michael/data/GIT/MachineLearning/data/forCART/Classifier')
	#print(obj._reader._xtrain)
	#obj._fixdata()
	#print(obj._reader._xtrain)
	obj.fit(max_depth=2,bool_prune=False)
	#obj.print_tree()
	#obj.save_model()
	#obj.load_model()
	obj.print_tree()
	#print('*************')
	#print(obj._cur_model)
	#print('*************')
	#obj._cur_model._root.showAttributes()
	#print(obj._stored_model)
	#验证集上预测结果
	#print(obj.eval(bool_use_stored_model=False)[0])
	#print(obj.eval(bool_use_stored_model=False)[1])
	#print('---')
	#for node in obj._cur_model.preOrder():
		#print(node)
	#	node.showAttributes()
	#print(obj.eval(bool_use_stored_model=False)[0])
	#print(obj.eval(bool_use_stored_model=False)[1])
	#print(obj.eval(bool_use_stored_model=True)[0])
	#print(obj.eval(bool_use_stored_model=True)[1])
	#验证集上评价结果
	#print(obj.eval(bool_use_stored_model=True,method='f1-score')[1])
	#执行预测
	#print(obj.predict([[0,0,0,0,0,0]]))
	#print(obj.predict([[10,10,10,10,10,10]]))
	#print(obj.predict([[1,1,1,1,1,0]],True))
	#obj.save_model()
	#obj.fit(alpha_leaf=0,max_depth=3,bool_prune=True)
	#obj.fit(alpha_leaf=0,bool_prune=False)
	#obj.print_tree()
	print(obj.eval(bool_use_stored_model=False)[0])
	print(obj.eval(bool_use_stored_model=False)[1])
	print(obj._cur_model._size)
