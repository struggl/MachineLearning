'''CART决策树实现回归'''
import numpy as np
import collections

from MLBase import DecisionTreeRegressorBase
class CARTRegressor(DecisionTreeRegressorBase):
	'''CART决策树回归，与C4.5决策树分类器实现的区别在于：
	1._chooseBestFeatureToSplit方法中不使用信息增益_calInformationGain而是误差平方和_calSSE
	2.save_model与load_model方法的默认路径
	'''
	def __init__(self,dataDir,reader=None):
		super().__init__(dataDir,reader)

	def _calScore(self,xtrain,ytrain,feat,splitVal):
		'''改写父类的同名方法,这里采用误差平方和指标(SSE)。与CART分类不同的是，CART回归时，分裂后SSE的计算不加权
		先根据特征feat和分割点spliVal将数据集一分为二，然后计算分割前后的指标增益'''	
		xtrain = self._assert_xdata(xtrain)
		ytrain = self._assert_ydata(ytrain)	

		left_data,right_data = self._splitDataSet(xtrain,ytrain,feat,splitVal)
		from math import log

		#计算数据集划分前后的mse指标收益
		baseSSE = self._calSSE(ytrain)
		newSSE = 0
		newSSE += self._calSSE(left_data[1]) + self._calSSE(right_data[1])
		SSEGain = baseSSE - newSSE
		
		return SSEGain
		
	def _chooseBestFeatureToSplit(self,xtrain,ytrain,epsion=0):
		'''使用误差平方和(Sum Square Error,SSE)选择最优划分特征,若数据集特征数不大于0或最优划分的指标收益
		不大于阈值epsion，则返回None
		Args:
			epsion:每次结点划分时损失函数下降的阈值，默认为0
		'''
		if epsion < 0:
			raise ValueError('结点分裂阈值epsion不能为负数!')
		numFeat = len(xtrain[0])
		if numFeat < 1:
			return None
		bestGain = epsion
		bestFeat = None
		bestSplitVal = None
		for feat in range(numFeat):
			splitValList = sorted(list(set(xtrain[:,feat])))
			for i in range(len(splitValList)-1):
				splitVal = (splitValList[i]+splitValList[i+1]) / 2.0				
				#划分后的指标增益，若为C4.5，则为信息增益比，若为CART回归树则为均方误差
				curGain = self._calScore(xtrain,ytrain,feat,splitVal)
				
				if curGain > bestGain:
					bestFeat = feat
					bestSplitVal = splitVal
					bestGain = curGain
		if bestFeat != None:
			return bestFeat,bestSplitVal,bestGain
	
	def _fixdata(self):
		self._reader._xtrain = np.asarray(self._reader._xtrain,dtype='float64')
		self._reader._xtest = np.asarray(self._reader._xtest,dtype='float64')
		self._reader._xeval = np.asarray(self._reader._xeval,dtype='float64')

	def _get_examples(self,node):
		'''获取node存放的样本'''
		self._cur_model._validate(node)
		return node._examples

	def _splitDataSet(self,xdata,ydata,bestFeat,bestSplitVal):
		'''根据最优特征的最优分割点将数据集一分为二'''
		xdata = self._assert_xdata(xdata)
		ydata = self._assert_ydata(ydata)	
		left_xdata = []
		left_ydata = []
		right_xdata = []
		right_ydata = []
		for i in range(len(xdata)):
			if xdata[i][bestFeat] <= bestSplitVal:
				left_xdata.append(xdata[i])
				left_ydata.append(ydata[i])
			else:
				right_xdata.append(xdata[i])
				right_ydata.append(ydata[i])
		left_xdata = np.asarray(left_xdata,dtype=xdata.dtype)	
		left_ydata = np.asarray(left_ydata,dtype=ydata.dtype)	
		
		right_xdata = np.asarray(right_xdata,dtype=xdata.dtype)
		right_ydata = np.asarray(right_ydata,dtype=ydata.dtype)
		return (left_xdata,left_ydata),(right_xdata,right_ydata)
		
	def _fit(self,xtrain,ytrain,examples,depth,max_depth,epsion):
		'''训练CART回归树。
		递归构建CART回归树的核心过程：
		遍历所有可能特征:
			对每个特征所有可能取值(连续值)进行排序，所有相邻取值的中点集合构成了所有可能的分割阈值
			遍历每个可能分割阈值进行二元划分，计算划分前后的指标
		最优指标对应的特征及最优分割阈值即为所求

		Args:
			epsion:float.默认为0.选取最优特征和最优分割点时，当前结点分裂前后指标(例如信息增益，基尼指数)
				变化量的最小阈值。
		'''
		if max_depth != None and depth > max_depth:	#递归返回情况1:树的深度达到最大设定时终止,返回None
			return

		freq = 0
		for lb in ytrain:
			if lb == ytrain[0]:
				freq += 1
		if freq == len(ytrain):	#递归返回情况2：所有样本因变量值相同,返回叶结点
			cur_node = self.Node(value=ytrain[0],
						examples=examples,
						depth=depth)
			cur_node._loss = 0
			if self._cur_model._root is None:
				self._cur_model.add_root(cur_node)
			return	cur_node
		
		#选择最优划分特征和最优划分阈值
		cur_loss = self._calSSE(ytrain)
		res = self._chooseBestFeatureToSplit(xtrain,ytrain,epsion=epsion)
		if res is None:	#递归返回情况3：无法继续切分特征时，返回叶结点
			cur_node = self.Node(value=np.mean(ytrain),
						examples=examples,
						loss=cur_loss,
						depth=depth)
			if self._cur_model._root is None:
				self._cur_model.add_root(cur_node)
			return	cur_node

		bestFeat,bestSplitVal,gain = res	
		resNode = self.Node(feature=bestFeat,
					splitVal=bestSplitVal,
					gain=gain,
					loss=cur_loss,
					examples=examples,
					depth=depth)

		if self._cur_model._root is None:
			self._cur_model.add_root(resNode)
		else:
			self._cur_model._size += 1
				
		#仅当当前结点深度depth小于限定深度max_depth时才分裂当前结点
		if max_depth is None or depth < max_depth:
			#根据最优特征和最优分割点为左右孩子划分数据集,返回的left_dataSet结构为(left_xtrain,left_ytrain),
			#right_dataSet类似
			left_dataSet,right_dataSet = self._splitDataSet(xtrain,ytrain,bestFeat,bestSplitVal)
			left = self._fit(xtrain=left_dataSet[0],
						ytrain=left_dataSet[1],
						examples=left_dataSet,
						depth=resNode._depth+1,
						max_depth=max_depth,
						epsion=epsion)
			right = self._fit(xtrain=right_dataSet[0],
						ytrain=right_dataSet[1],
						examples=right_dataSet,
						depth=resNode._depth+1,
						max_depth=max_depth,
						epsion=epsion)
			#父结点指向对应孩子
			resNode._left = left
			resNode._right = right

			#维护孩子的_parent属性
			left._parent = resNode
			right._parent = resNode

			#维护孩子的_parent_split_feature_val属性
			left._parent_split_feature_val = (bestFeat,bestSplitVal,'left')			
			right._parent_split_feature_val = (bestFeat,bestSplitVal,'right')			
			
			#维护决策树结点数量属性，C4.5决策树的建树过程保证了只要当前结点能分裂，则必有两个孩子
			self._cur_model._size += 2

		#若当前结点未分裂(深度到达限制),需要设定当前结点为叶结点
		if self._cur_model.num_children(resNode) == 0:
			resNode._feature = None
			resNode._value = np.mean(ytrain)	
		return resNode	

	#-------------------------------------------------公开接口------------------------------------------------
	def fit(self,xtrain=None,
			ytrain=None,
			examples=None,
			depth=None,
			max_depth=None,
			alpha_leaf=0,
			bool_prune=False,
			epsion=0.0):
		"""模型拟合的公开接口。若训练数据集未直接提供，则使用self._reader读取训练数据集
		Args:
			alpha_leaf:后剪枝对叶结点的正则化超参数,有效取值大于等于0.
			epsion:float.默认为0.选取最优特征和最优分割点时，当前结点分裂前后指标(例如信息增益，基尼指数)
				变化量的最小阈值。
		"""
		if xtrain is None or ytrain is None:
			self._fixdata()
			self._cur_model = self.DecisionTree()
			self._fit(xtrain=self._reader._xtrain,
					ytrain=self._reader._ytrain,
					examples=(self._reader._xtrain,self._reader._ytrain),
					depth=1,
					max_depth=max_depth,
					epsion=epsion)

		else:
			xtrain = self._assert_xdata(xtrain)
			ytrain = self._assert_ydata(ytrain)
			self._cur_model = self.DecisionTree()
			self._fit(xtrain=self._reader._xtrain,
					ytrain=self._reader._ytrain,
					examples=(self._reader._xtrain,self._reader._ytrain),
					depth=1,
					max_depth=max_depth,
					epsion=epsion)
		if bool_prune:
			self._prune(alpha_leaf=alpha_leaf)	

	def bool_not_trained(self,tree=None):
		'''判断决策树是否已经训练,仅判断根结点，默认在_fit和_prune方法的更新过程中其余结点维护了相应的特性'''
		if tree is None:
			tree = self._cur_model
		if tree is None or tree._root is None:
			return True
		if tree._root._left is None and tree._root._right is None and tree._root._value is None:
			return True
		return False	

	def predict(self,xtest=None,bool_use_stored_model=False):
		'''模型预测的公开接口'''
		if bool_use_stored_model:
			use_model = self._stored_model
			
		else:
			use_model = self._cur_model
		if self.bool_not_trained(use_model):
			raise self.NotTrainedError('无法进行预测，因为决策树分类器尚未训练!')

		if xtest is None:
			cur_xtest = self._reader.xtest
		else:
			cur_xtest = self._assert_xdata(xtest)

		if use_model.is_leaf(use_model._root):
			preds = [use_model._root._value] * len(cur_xtest)
			return np.asarray(preds)

		preds = [None] * len(cur_xtest) 
		for i in range(len(cur_xtest)):
			node = use_model._root
			while not use_model.is_leaf(node):
				if cur_xtest[i][node._feature] <= node._splitVal:
					node = node._left
				else:
					node = node._right
			preds[i] = node._value	
		return np.asarray(preds)

	def eval(self,bool_use_stored_model=False,method=None):
		preds = self.predict(self._reader._xeval,bool_use_stored_model)	
		return preds,self._evaluator.eval(preds,self._reader._yeval,method) 

	def save_model(self,path=None):
		'''决策树分类器序列化'''
		if self.bool_not_trained():
			raise self.NotTrainedError('无法进行模型序列化，因为决策树分类器尚未训练!')
		if path is None:
			cur_path = self._reader._dataDir + '/CARTRegressor.pkl'
		else:
			cur_path = path
		import pickle
		with open(cur_path,'wb') as f:
			pickle.dump(self._cur_model,f)
		print('save_model done!')
	
	def load_model(self,path=None):
		'''载入模型'''
		if path is None:
			cur_path = self._reader._dataDir + '/CARTRegressor.pkl'
		else:
			cur_path = path
		import pickle
		with open(cur_path,'rb') as f:
			self._stored_model = pickle.load(f)
		print('load_model done!')

	def print_tree(self):
		'''层序遍历输出决策树结点及结点关键信息'''
		if self._cur_model is None:
			use_model = self._stored_model
		else:
			use_model = self._cur_model

		Q = collections.deque()
		Q.append(use_model._root)
		while len(Q) != 0:
			node = Q.popleft()
			node.showAttributes()
			print('---\n')
			for child in use_model.children(node):
				Q.append(child)

	class Node(DecisionTreeRegressorBase.Node):
			'''决策树的结点类'''
			__slots__ = '_feature','_value','_left','_right','_parent','_depth',\
					'_examples','_parent_split_feature_val','_splitVal','_gain'
			def __init__(self,feature=None,
						value=None,
						left=None,
						right=None,
						parent=None,
						depth=None,
						examples=None,
						splitVal=None,
						loss=None,
						gain=None):
				'''由于C4.5采用二元划分对连续属性进行分割，因此C4.5结点定义的属性与ID3有所区别
				Args:
					feature:存储当前结点的划分属性
					value:若为叶结点，则_value属性存储了该叶结点所拥有的ydata的均值,否则为None
					left:当前结点的左孩子
					right:当前结点的右孩子
					parent父结点，根结点设置为None
					depth:结点的深度,根结点设置深度为1
					examples:tuple.每个结点存储了自己拥有的xtrain与ytrain
					splitVal:float.存储了本结点分裂特征的分割阈值
					gain:存储当前最优分裂结点对应的指标增益(典型指标函数为信息增益、信息增益比、基尼指数)
					loss:存储当前结点ydata的误差平方和(SSE)
				'''
				self._feature = feature
				self._value = value
				self._left = left
				self._right = right
				self._parent = parent	
				self._depth = depth
				self._examples = examples
				self._splitVal = splitVal
				self._loss = loss
				self._gain = gain
				#存储父结点的分裂特征及分割点取值
				#格式为(bestFeat,bestSplitVal,'right')或(bestFeat,bestSplitVal,'left')			
				self._parent_split_feature_val = None
		
			def showAttributes(self):
				print('_depth:'+repr(self._depth))
				print('父结点划分特征取值_parent_split_feature_val:'+repr(self._parent_split_feature_val))
				print('当前划分属性_feature:'+repr(self._feature))
				print('当前结点划分阈值_splitVal:'+repr(self._splitVal))
				print('_gain:'+repr(self._gain))
				print('_loss:'+repr(self._loss))
				print('_value:'+repr(self._value))
				#print('_examples:'+repr(self._examples))

	class DecisionTree(DecisionTreeRegressorBase.DecisionTree):
		'''决策树数据结构'''
		def __init__(self):
			self._size = 0
			self._root = None

		def __len__(self):
			return self._size

		def _validate(self,node):
			if not isinstance(node,DecisionTreeRegressorBase.Node):
				raise TypeError

		def is_leaf(self,node):
			self._validate(node)
			return node._left is None and node._right is None and node._value != None

		def is_root(self,node):
			self._validate(node)
			return self._root is node

		#-----------------------------访问方法-----------------------------
		def preOrder(self,node=None):
			'''从node开始进行前序遍历，若node为None，则从根开始遍历,返回一个迭代器'''
			if node is None:
				node = self._root
			if isinstance(node,DecisionTreeRegressorBase.Node):
				yield node
				for child in self.children(node):
					for nd in self.preOrder(child):
						yield nd

		def parent(self,node):
			'''返回给定node的父结点'''
			self._validate(node)
			return node._parent

		def children(self,node):
			'''返回给定结点node的孩子结点的迭代器'''
			self._validate(node)
			if node._left != None:
				yield node._left
			if node._right != None:
				yield node._right
		
		def sibling(self,node):
			'''返回给定结点node的兄弟结点的迭代器'''
			self._validate(node)
			if node is node._parent._left:
				return node._parent._right
			else:
				return node._parent._left

		def num_children(self,node):
			self._validate(node)
			num = 0
			if node._left != None:
				num += 1
			if node._right != None:
				num += 1
			return num
 
		#-----------------------------更新方法------------------------------				
		def add_root(self,node):
			'''为决策树添加根结点，根结点深度设定为1'''
			self._root = node
			node._depth = 1
			self._size = 1


if __name__ == '__main__':
	obj = CARTRegressor(dataDir='/home/michael/data/GIT/MachineLearning/data/forCART/Regressor')
	#print(obj._reader._xtrain)
	#obj._fixdata()
	#print(obj._reader._xtrain)
	obj.fit(max_depth=7,bool_prune=False)
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
	print(len(obj._cur_model))
