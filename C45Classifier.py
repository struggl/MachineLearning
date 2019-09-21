'''ID3决策树实现'''
import numpy as np
import collections

from MLBase import DecisionTreeClassifierBase
class C45Classifier(DecisionTreeClassifierBase):
	'''C4.5决策树分类器，与ID3决策树实现的区别在于：
	1._chooseBestFeatureToSplit方法中不使用信息增益_calInformationGain而是信息增益比_calInformationGainRatio
	2._fixdata方法需要保证特征向量全部为float型
	'''
	def _calInformationGainRatio(xtrain,ytrain,feat,splitVal):
		'''改写父类的同名方法,先根据特征feat和分割点spliVal将数据集一分为二，然后计算分割前后的指标增益'''	
	

	def _chooseBestFeatureToSplit(self,xtrain,ytrain,epsion=0):
		'''使用信息增益比选择最优划分特征,若数据集特征数不大于0或最优划分的信息增益比大于阈值epsion，则返回None
		Args:
			epsion:每次结点划分时损失函数下降的阈值，默认为0
		'''
		if epsion < 0:
			raise ValueError('结点分裂阈值epsion不能为负数!')
		numFeat = len(xtrain[0])
		if numFeat < 1:
			return None
		bestGainRatio = epsion
		bestFeat = None
		bestSplitVal = None
		for feat in range(numFeat):
			splitValList = sorted(list(set(xtrain[:feat])))
			for i in range(len(splitValList)-1):
				splitVal = (splitValList[i]+splitValList[i+1]) / 2.0				
				curGainRatio = self._calInformationGainRatio(xtrain,ytrain,feat,splitVal)
				if curGainRatio > bestGainRatio:
					bestFeat = feat
					bestSplitVal = splitVal
					bestGainRatio = curGainRatio

		if bestFeat != None and bestGainRatio > epsion:
			return bestFeat,bestSplitVal,bestGainRatio
	
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
		
	def _fit(self,xtrain,ytrain,examples,depth,max_depth,epsion=0):
		'''训练C4.5决策树分类器
		ID3每个结点的可能子树数量为指定特征的取值数量，而C4.5每个结点的可能子树数量恒为2。
		
		C4.5递归构建决策树的核心过程：
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
		if freq == len(ytrain):	#递归返回情况2：所有类别相同,返回叶结点
			cur_node = self.Node(label=ytrain[0],
						examples=examples,
						depth=depth)
			if self._cur_model._root is None:
				self._cur_model.add_root(cur_node)
			return	cur_node
		
		#选择最优划分特征和最优划分阈值
		res = self._chooseBestFeatureToSplit(xtrain,ytrain,epsion=epsion)
		if res is None:	#递归返回情况3：无法继续切分特征时，返回叶结点
			cur_node = self.Node(label=self._majority_class(ytrain),
						examples=examples,
						depth=depth)
			if self._cur_model._root is None:
				self._cur_model.add_root(cur_node)
			return	cur_node

		bestFeat,bestSplitVal,loss = res	
		resNode = self.Node(feature=bestFeat,
					splitVal=splitVal,
					loss=loss,
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
						max_depth=max_depth)
			right = self._fit(xtrain=right_dataSet[0],
						ytrain=right_dataSet[1],
						examples=right_dataSet,
						depth=resNode._depth+1,
						max_depth=max_depth)
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
			resNode._label = self._majority_class(ytrain)	
		return resNode	


	def eval(self,bool_use_stored_model=False,method=None):
		preds = self.predict(self._reader._xeval,bool_use_stored_model)	
		return preds,self._evaluator.eval(preds,self._reader._yeval,method) 

	def save_model(self,path=None):
		'''决策树分类器序列化'''
		if self.bool_not_trained():
			raise self.NotTrainedError('无法进行模型序列化，因为决策树分类器尚未训练!')
		if path is None:
			cur_path = self._reader._dataDir + '/C45Classifier.pkl'
		else:
			cur_path = path
		import pickle
		with open(cur_path,'wb') as f:
			pickle.dump(self._cur_model,f)
		print('save_model done!')
	
	def load_model(self,path=None):
		'''载入模型'''
		if path is None:
			cur_path = self._reader._dataDir + '/C45Classifier.pkl'
		else:
			cur_path = path
		import pickle
		with open(cur_path,'rb') as f:
			self._stored_model = pickle.load(f)
		print('load_model done!')

	class Node(DecisionTreeClassifierBase.Node):
			'''决策树的结点类'''
			__slots__ = '_feature','_label','_left','_right','_parent','_depth',\
					'_examples','_parent_split_feature_val','_splitVal','_loss'
			def __init__(self,feature=None,
						label=None,
						left=None,
						right=None,
						parent=None,
						depth=None,
						examples=None,
						splitVal=None,
						loss=None):
				'''由于C4.5采用二元划分对连续属性进行分割，因此C4.5结点定义的属性与ID3有所区别
				Args:
					feature:存储当前结点的划分属性
					label:若为叶结点，则_label属性存储了该叶结点的标签,否则为None
					left:当前结点的左孩子
					right:当前结点的右孩子
					parent父结点，根结点设置为None
					depth:结点的深度,根结点设置深度为1
					examples:tuple.每个结点存储了自己拥有的xtrain与ytrain
					splitVal:float.存储了本结点分裂特征的分割阈值
					loss:存储当前最优分裂结点对应的损失值(典型损失函数为信息增益、信息增益比、基尼指数)

				'''
				self._feature = feature
				self._label = label
				self._left = left
				self._right = right
				self._parent = parent	
				self._depth = depth
				self._examples = examples
				self._splitVal = splitVal
				self._loss = loss
				#存储父结点的分裂特征及分割点取值
				#格式为(bestFeat,bestSplitVal,'right')或(bestFeat,bestSplitVal,'left')			
				self._parent_split_feature_val = None
		
			def showAttributes(self):
				print('_depth:'+repr(self._depth))
				print('父结点划分特征取值_parent_split_feature_val:'+repr(self._parent_split_feature_val))
				print('当前划分属性_feature:'+repr(self._feature))
				print('当前结点划分阈值_cur_split_val:'+repr(self._cur_split_val))
				print('_loss:'+repr(self._loss))
				print('_label:'+repr(self._label))

	class DecisionTree(DecisionTreeClassifierBase.DecisionTree):
		'''决策树数据结构'''
		def __init__(self):
			self._size = 0
			self._root = None

		def __len__(self):
			return self._size

		def _validate(self,node):
			if not isinstance(node,DecisionTreeClassifierBase.Node):
				raise TypeError

		def is_leaf(self,node):
			self._validate(node)
			return node._left is None and node._right is None and node._label != None

		def is_root(self,node):
			self._validate(node)
			return self._root is node

		#-----------------------------访问方法-----------------------------
		def preOrder(self,node=None):
			'''从node开始进行前序遍历，若node为None，则从根开始遍历,返回一个迭代器'''
			if node is None:
				node = self._root
			if isinstance(node,DecisionTreeClassifierBase.Node):
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
	obj = C45Classifier(dataDir='/home/michael/data/GIT/MachineLearning/data/forID3')
	print(obj._reader._xtrain)
	obj._fixdata()
	print(obj._reader._xtrain)
	#obj.fit(alpha_leaf=0.55,bool_prune=True)
	#obj.print_tree()
	#obj.save_model()
	#obj.load_model()
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
	obj.fit(alpha_leaf=0,bool_prune=False)
	obj.print_tree()
	print(obj.eval(bool_use_stored_model=False)[0])
	print(obj.eval(bool_use_stored_model=False)[1])
	print(obj._cur_model._size)
