from abc import ABCMeta,abstractmethod
import collections
import numpy as np

class LearnerBase(metaclass=ABCMeta):
	"""所有学习器的基类"""
	class NotTrainedError(Exception):
		pass
		
	class PredictionError(Exception):
		pass
	
	@abstractmethod
	def fit(self,xtrain,ytrain):
		pass

	@abstractmethod
	def predict(self,xtest):
		pass
	
	@abstractmethod
	def eval(self,predicts,labels,method=None):
		pass

	@abstractmethod
	def load_model(self,path):
		pass

	@abstractmethod
	def save_model(self,path):
		pass
	
	def getLearnerType(self):
		return self._LearnerType


class EvaluatorBase(metaclass=ABCMeta):
	"""所有模型评价指标的基类"""
	__slots__ = '_smallDigit','_evaluatorType'
	def __init__(self,smallDigit=None):
		self._smallDigit = smallDigit if smallDigit is not None else 0.000001
		self._evaluatorType = None

	@abstractmethod
	def get_all_evaluation_method(self):
		'''获取所有支持的评价方法'''
		pass	
	
	@abstractmethod
	def eval(self,predicts,labels,method=None):
		'''根据模型预测与数据指标进行模型评测
		Args:
			predicts:模型预测结果
			labels:数据原来的标签
			method:str.指定使用的评价方法
		'''
	def getEvaluatorType(self):
		return self._evaluatorType


class ClassifierEvaluator(EvaluatorBase):
	"""分类模型的评价器"""
	def __init__(self,smallDigit=None):
		super().__init__(smallDigit)
		self._evaluatorType = 'Classifier'

	def _getAccuracy(self,preds,labels):
		'''计算准确率'''
		assert type(preds) == np.ndarray and type(labels) == np.ndarray
		assert preds.dtype == 'int64' or preds.dtype == 'int32' 
		assert labels.dtype == 'int64' or labels.dtype == 'int32' 
		assert len(preds) != 0 and len(preds) == len(labels)

		return sum(preds==labels) / len(preds)	
	
	def _getRecall(self,preds,labels):
		'''计算各类别的召回，返回一个字典'''
		res = {}
		lbs = set(labels)
		for lb in lbs:
			num_lb = 0
			num_right = 0
			for i in range(len(preds)):
				if labels[i] == lb:
					num_lb += 1
					if preds[i] == lb:
						num_right += 1
			#这里的num_lb不需要加上一个极小数
			res[lb] = num_right / num_lb
		return res	

	def _getPrecision(self,preds,labels):
		'''计算各类别的精度，返回一个字典'''
		res = {}
		lbs = set(labels)
		for lb in lbs:
			num_lb = 0
			num_right = 0
			for i in range(len(preds)):
				if preds[i] == lb:
					num_lb += 1
					if labels[i] == lb:
						num_right += 1
			#注意，对于特定类别，模型预测为该类的样本数量可能为0
			res[lb] = num_right / (num_lb+self._smallDigit)
		return res	

	def _getF1score(self,preds,labels):
		'''计算各类别的f1-score，返回一个字典'''
		P = self._getPrecision(preds,labels)
		R = self._getRecall(preds,labels)
		F1 = dict()
		#注意，对于一个特定的类，P与R可能都取0
		for k in P.keys():
			F1[k] = 2*P[k]*R[k] / (P[k]+R[k]+self._smallDigit)	
		return F1		
			
	#----------------------------------公开接口----------------------------------
	def get_all_evaluation_method(self):
		'''获取所有支持的评价方法'''
		return {'accuracy','precision','recall','f1-score'}
		
	def eval(self,predictions,labels,method=None):
		'''指定方法对模型结果进行评价
		Args:
			method:str.可选值在get_all_evaluation_method返回的集合中
		'''		
		if method is None or method == 'f1-score':
			return self._getF1score(predictions,labels)	
	
		if method not in self.get_all_evaluation_method():
			raise ValueError('method参数仅支持以下取值: '+repr(self.get_all_evaluation_method()))

		if method == 'accuracy':
			return self._getAccuracy(predictions,labels)
		if method == 'precision':
			return self._getPrecision(predictions,labels)
		if method == 'recall':
			return self._getRecall(predictions,labels)

class RegressorEvaluator(EvaluatorBase):
	"""回归模型的评价器"""
	def __init__(self):
		self._EvaluatorType = 'Regressor'


class ClassifierBase(LearnerBase):
	"""所有分类器的基类"""
	__slots__ = '_learnerType','_evaluator','_reader','_cur_model','_stored_model'
	def __init__(self,dataDir,reader=None):
		self._learnerType = 'Classifier'
		self._evaluator = ClassifierEvaluator()	
		self._reader = reader if reader is not None else tsvReader(dataDir)
		self._reader.read()
		self._reader.transformLabelToInt64()
		self._cur_model = None
		self.stored_model = None


class RegressorBase(LearnerBase):
	"""所有回归器的基类"""
	def __init__(self):
		self._LearnerType = 'Classifier'
		self._Evaluator = RegressorEvaluator()	


class ReaderBase(metaclass=ABCMeta):
	"""数据读取器基类,读取器对象存储数据，并提供数据预处理的方法"""
	__slots__ = '_dataDir'
	
	def __init__(self,dataDir=None):
		self._dataDir = dataDir
	
	@abstractmethod
	def read(self):
		pass

	
class tsvReader(ReaderBase):
	'''tsv格式的读取器'''
	def __init__(self,dataDir):
		super().__init__(dataDir)

	def _read(self,path):
		'''读取数据为np.ndarray,且dtype为float64,默认最后一列为标签列'''
		fr = open(path,'r',encoding='utf-8')
		x = []
		y = []
		for line in fr:
			try:
				example = [float(feat) for feat in line.strip().split('\t')]
				x.append(example[:-1])
				y.append(example[-1])
			except ValueError:
				print('------{}------'.format(path))
		return np.asarray(x),np.asarray(y)	
		
	def read(self):	
		self._xtrain,self._ytrain = self._read(self._dataDir+'/train.tsv')
		self._xtest,self._ytest = self._read(self._dataDir+'/test.tsv')
		try:
			self._xeval,self._yeval = self._read(self._dataDir+'/eval.tsv')
		except FileNotFoundError:
			self._xeval,self._yeval = self._xtest,self._ytest 

	def transformLabelToInt64(self):
		if type(self._ytrain) != None:
			self._ytrain = np.asarray(self._ytrain,dtype='int64')	
		if type(self._ytest) != None:
			self._ytest = np.asarray(self._ytest,dtype='int64')	
		if type(self._yeval) != None:
			self._yeval = np.asarray(self._yeval,dtype='int64')	
			

class DecisionTreeClassifierBase(ClassifierBase):
	'''决策树分类器基类'''
	class Node(object):
		'''决策树的结点类'''
		def __init__(self,feature=None,label=None,examples=None,parent=None,depth=None):
			'''
			Args:
				feature:存储当前结点的划分属性
				label:若为叶结点，则_label属性存储了该叶结点的标签
				examples:每个结点存储了自己拥有的样本的序号(可迭代对象)
				parent父结点，根结点设置为None
				depth:结点的深度
			'''
			self._feature = feature
			self._children = collections.OrderedDict()
			#若为叶结点，则_label属性存储了该叶结点的标签
			self._label = label
			self._examples = examples
			self._parent = parent	
			self._depth = depth
	
		def showAttributes(self):
			print('_feature:'+repr(self._feature))
			print('_children:'+repr(self._children))
			print('_label:'+repr(self._label))
			print('_examples:'+repr(self._examples))
			print('_parent:'+repr(self._parent))
			print('_depth:'+repr(self._depth))
	
	class DecisionTree(object):
		'''决策树数据结构'''
		def __init__(self):
			self._size = 0
			self._root = None
			self._depth = None

		def __len__(self):
			return self._size

		def _validate(self,node):
			if not isinstance(node,DecisionTreeClassifierBase.Node):
				raise TypeError

		def is_leaf(self,node):
			self._validate(node)
			return node._children == {} and node._label != None

		def is_root(self,node):
			self._validate(node)
			return self._root is node

		#-----------------------------访问方法-----------------------------
		def preorder(self,node=None):
			'''从node开始进行前序遍历，若node为None，则从根开始遍历,返回一个迭代器'''
			if node is None:
				node = self._root
			if isinstance(node,DecisionTreeClassifierBase.Node):
				yield node
				if not self.is_leaf(node):
					for child in self.children(node):
						for nd in self.preorder(child):
							yield nd

		def parent(self,node):
			'''返回给定node的父结点'''
			self._validate(node)
			return node._parent

		def children(self,node):
			'''返回给定结点node的孩子结点的迭代器'''
			self._validate(node)
			for v in node._children.values():
				yield v
		
		def sibling(self,node):
			'''返回给定结点node的兄弟结点的迭代器'''
			self._validate(node)
			for v in node._parent._children.values():
				if v is not node:
					yield v

		def num_children(self,node):
			self._validate(node)
			if self.is_leaf(node):
				return 0
			else:
				return len(node._children)
			
 
		#-----------------------------更新方法------------------------------				
		def add_root(self,feature=None,label=None,examples=None,parent=None,depth=None):
			T._root = Node(feature=feature,label=label,examples=examples,parent=parent,depth=0)
			T._depth = 0
			T._size = 1

		def add_children(self,parent_node,key,feature=None,label=None,examples=None,parent=None,depth=None):
			'''根据key为parent_node添加孩子child'''
			self._validate(parent_node)
			child = Node(feature=feature,label=label,examples=examples,parent=parent,depth=parent_node._depth+1)
			parent_node._children[key] = child
			child._parent = parent_node
			T._size += 1
			T._depth = max(T._depth,child._depth)

	def bool_not_trained(self,tree=None):
		'''判断决策树是否已经训练,仅判断根结点，默认在_fit和_prune方法的更新过程中其余结点维护了相应的特性'''
		if tree is None:
			tree = self._cur_model
		if tree is None or tree._root is None:
			return True
		if tree._root._children == {} and tree._root._label is None:
			return True
		return False
		
	def _assert_xdata(self,xdata):
		assert type(xdata) is np.ndarray
		assert xdata.ndim == 2 
		assert len(xdata) != 0

	def _assert_ydata(self,ydata):
		assert type(ydata) is np.ndarray
		assert ydata.ndim == 1
		assert ydata.dtype == 'int64' or ydata.dtype == 'int32'
		assert len(ydata) != 0

	def _majority_class(self,ytrain):
		ytrain = np.asarray(ytrain)
		self._assert_ydata(ytrain)
		freq = {}
		for lb in ytrain:
			freq[lb] = freq.get(lb,0) + 1	
		return sorted(freq.items(),key=lambda x:x[1],reverse=True)[0][0]

	def _calInformationEntropy(self,xdata,ydata):
		'''给定数据集D，计算数据集D的信息熵,信息熵取值越小越好.令K为类别数量，
		Ent(D) = - sum_{k=1}^K p_k*log(p_k,2),其中p_k为第k类在数据集中出现的频率
		'''
		xdata = np.asarray(xdata)
		ydata = np.asarray(ydata)
		self._assert_xdata(xdata)
		self._assert_ydata(ydata)
		from math import log
		totalEnt = 0.0
		lbFreq = {}
		for lb in ydata:
			lbFreq[lb] = lbFreq.get(lb,0) + 1
		nexample = len(xdata)	
		for k in lbFreq.keys():
			p_k = float(lbFreq[k]) / nexample
			totalEnt -= p_k * log(p_k,2) 
		return totalEnt
			
	def _calGini(self,xdata,ydata):
		'''给定数据集D，计算数据集D的基尼指数，基尼指数取值越小越好。令K为类别数量，
		Gini(D) = 1 - sum_{k=1}^K p_k^2,其中p_k为第k类在数据集中出现的频率
		'''	
		xdata = np.asarray(xdata)
		ydata = np.asarray(ydata)
		self._assert_xdata(xdata)
		self._assert_ydata(ydata)
		lbFreq = {}
		for lb in ydata:
			lbFreq[lb] = lbFreq.get(lb,0) + 1
		nexample = len(xdata)	
		Gini = 0
		for k in lbFreq.keys():
			p_k = float(lbFreq[k]) / nexample
			Gini += p_k ** 2
		Gini = 1 - Gini
		return Gini

	def _calInformationGain(self,xdata,ydata,feat):
		'''给定数据集D和属性feat，计算属性feat的信息增益.
		假设属性feat上取值为{a_1,...,a_v,...a_V},对应划分得到的子数据集(不含feat这一列)为{D^1,...,D^v,...,D^V}则
		Gain(D,a) = Ent(D) - sum_{v=1}^V p_v*Ent(D^v),其中p_v表示属性取值a_v在原数据集中的数量占比 
		'''
		xdata = np.asarray(xdata)
		ydata = np.asarray(ydata)
		self._assert_xdata(xdata)
		self._assert_ydata(ydata)
		baseInformationEntropy = self._calInformationEntropy(xdata,ydata)
		featValSet = set([example[feat] for example in xdata])
		nexample = float(len(xdata))
		newInformationEntropy = 0.0
		for val in featValSet:
			cur_xdata,cur_ydata = self._splitDataSet(xdata,ydata,feat,val,False)
			p_v = len(cur_xdata) / nexample
			newInformationEntropy += p_v * self._calInformationEntropy(cur_xdata,cur_ydata)
		InformationGain = baseInformationEntropy - newInformationEntropy
		return InformationGain	
												
	def _calInformaitonGainRatio(self,xdata,ydata,feat):
		'''给定数据集D和属性feat，计算属性feat的信息增益率'''
		xdata = np.asarray(xdata)
		ydata = np.asarray(ydata)
		self._assert_xdata(xdata)
		self._assert_ydata(ydata)
		pass

	def _calGiniIndex(self,xdata,ydata,feat):
		'''给定数据集D和属性feat，计算属性feat的基尼指数'''
		xdata = np.asarray(xdata)
		ydata = np.asarray(ydata)
		self._assert_xdata(xdata)
		self._assert_ydata(ydata)
			
	def _splitDataSet(self,xtrain,ytrain,feat,val,
				bool_return_examples=True,
				bool_contain_feat_column=False):
		'''获取子数据集feat列上取值为val的子数据集
		Args:
			xtrain:np.ndarray,二维特征向量
			ytrain:np.ndarray,维度是一。
			feat:python int.指定列索引
			val:python int.指定feat列的取值
			bool_return_examples:bool.若为True,返回xtrain的feat列上取值为val的那些样本索引的可迭代对象(且支持__len__)
			bool_contain_feat_column:bool.若为True,则采用CART决策树的二元划分策略，子数据集仍然包含feat这一列;
				否则子数据集不保留feat所在列。
		'''
		if bool_contain_feat_column:
			raise NotImplementedError
		xtrain = np.asarray(xtrain)
		ytrain = np.asarray(ytrain)
		self._assert_xdata(xtrain)
		self._assert_ydata(ytrain)
		xdata = []
		ydata = []
		examples = set()
		for i in range(len(xtrain)):
			if xtrain[i][feat] == val:
				examples.add(i)
				xdata.append(np.concatenate([xtrain[i][:feat],xtrain[i][feat+1:]]))
				ydata.append(ytrain[i])
		xdata,ydata = np.asarray(xdata,dtype=xtrain.dtype),np.asarray(ydata,dtype=ytrain.dtype)
		if bool_return_examples:
			return xdata,ydata,examples
		else:
			return xdata,ydata

	
class ID3Classifier(DecisionTreeClassifierBase):
	'''ID3决策树分类器'''	
	def __init__(self,dataDir,reader=None):
		super().__init__(dataDir,reader)
	
	def _chooseBestFeatureToSplit(self,xtrain,ytrain):
		'''使用信息熵选择最优划分特征,若数据集特征数不大于1或最优划分的信息增益不为正，则返回None
		'''
		numFeat = len(xtrain[0])
		if numFeat <= 1:
			return None
		bestGain = 0.0
		bestFeat = None
		for feat in range(numFeat):
			curGain = self._calInformationGain(xtrain,ytrain,feat)
			if curGain > bestGain:
				bestGain = curGain
				bestFeat = feat
		if bestFeat != None and bestGain > 0:
			return bestFeat
		return None	

	def _fit(self,xtrain,ytrain,examples,depth,max_depth=None):
		'''训练决策树分类器
		'''
		freq = 0
		for lb in ytrain:
			if lb == ytrain[0]:
				freq += 1
		if freq == len(ytrain):	#递归返回情况1：所有类别相同
			return self.Node(label=ytrain[0],
					examples=examples,
					depth=depth+1)

		bestFeat = self._chooseBestFeatureToSplit(xtrain,ytrain)	#选择最优划分特征
		if bestFeat is None:	#递归返回情况2：无法继续切分特征时，返回众数类
			return self.Node(label=self._majority_class(ytrain),
					examples=examples,
					depth=depth+1)
		
		bestFeatVals = set([example[bestFeat] for example in xtrain])
		
		resNode = self.Node(feature=bestFeat,examples=examples,depth=depth+1)
	
		for val in bestFeatVals:	#对最优特征的每个值构建子树	
			#_splitDataSet方法需要新增一个返回，记录xtrain中bestFea等于val的那些行
			cur_xtrain,cur_ytrain,cur_examples = self._splitDataSet(xtrain,ytrain,bestFeat,val)
			newChild = self._fit(xtrain=cur_xtrain,
						ytrain=cur_ytrain,
						examples=cur_examples,
						depth=resNode._depth+1)
			resNode._children[val] = newChild
			newChild._parent = resNode
		return resNode	

	def _fixdata(self):
		self._reader._xtrain = np.asarray(self._reader._xtrain,dtype='int64')
		self._reader._xtest = np.asarray(self._reader._xtest,dtype='int64')
		self._reader._xeval = np.asarray(self._reader._xeval,dtype='int64')

	def _get_examples(self,node):
		'''获取node存放的样本'''
		self._cur_model._validate(node)
		n = len(node._examples)
		xdata = [None] * n
		ydata = [None] * n
		i = 0
		for idx in node._examples:
			xdata[i] = self._reader._xtrain[idx]
			ydata[i] = self._reader._ytrain[idx]
			i += 1
		return np.asarray(xdata),np.asarray(ydata)
	
	def _loss(self,node):
		xdata,ydata = self._get_examples(node)
		return self._calInformationEntropy(xdata,ydata)

	#def bool_not_trained(self,tree=None):
	#	if tree is None:
	#		tree = self._cur_model
	#	if tree is None or tree._root is None:
	#		return True
	#	if tree._root._children == {} and tree._root._label is None:
	#		return True
	#	return False
				

	def _prune(self,alpha_leaf):
		'''决策树模型后剪枝的实现
		首先，对于同一个父结点的所有叶结点，是否满足剪枝条件的结论是一致的。记结点nd的兄弟结点为集合sibling
		而实现过程中比较麻烦的情况：
		1.某叶结点nd若可以上提叶结点(即令其父结点为叶结点)，则遍历到nd的sibling时，剪枝前叶结点数量len_before
			变量需要维护，这个问题通过new_leafs的过滤解决了
		2.若某叶结点nd不满足剪枝条件，当遍历到nd的sibling时，公式上需要考虑已经出队了的nd的损失计算和维护len_before
			变量为合理值,这个问题通过arrived_leafs的过滤解决了。解决方案中并没有遍历计算nb和sibling的损失，因为
			遍历到sibling时，肯定不会剪枝，而遍历到其他叶结点时，对于loss_before和loss_after，nb和sibling的损失
			增量是相等的(无论是对信息熵还是叶结点数量的正则项而言都相等)，因此无需计算

		Args:
			alpha_leaf:后剪枝对叶结点的正则化超参数,有效取值大于等于0.
		'''
		if self.bool_not_trained():
			raise self.NotTrainedError('无法进行剪枝，因为决策树分类器尚未训练!')
		if alpha_leaf < 0:
			raise ValueError('alpha_leaf必须非负!')

		#遍历决策树，把所有的叶结点入队列
		leafs = collections.deque()
		for nd in self._cur_model.preorder():
			if self._cur_model.is_leaf(nd):
				leafs.append(nd)

		#初始化叶结点数量为原树的叶结点个数
		len_before = len(leafs)
		#记录由于剪枝而新产生的叶结点
		new_leafs = set()
		#记录while遍历过，但没有进行剪枝的叶结点(同个父结点的叶结点，只需要记录一个就够了)
		arrived_leafs = set()	

		#除了队列不能为空以外，还需根结点不在叶结点集合中,否则有可能把原树删成空树
		while len(leafs) != 0 and self._cur_model._root not in leafs:
			#出队列,记为nd
			nd = leafs.popleft()
			#若nd的父结点已经在new_leafs中，则说明nd已经被剪除，直接下一次迭代即可
			if nd._parent in new_leafs:
				continue
			#若nd的兄弟结点在arrived_leafs中，则说明nd也不会达成剪枝条件，直接下一次迭代即可
			#同时因为没有剪枝，因此sibling方法仍然能正确返回nd的兄弟结点
			for lf in self._cur_model.sibling(nd):
				if lf in arrived_leafs:
					len_before -= 1
					continue
		
			#暂且把nd的父结点设定为叶结点
			new_leafs.add(nd._parent)
			#假设剪枝，则剪枝后的叶结点数量为原来的数量减去nd父结点的孩子数量，然后加1
			len_after = len_before - self._cur_model.num_children(nd._parent) + 1
			#剪枝前的损失计算初始化：
			#1.由于nd已经出队，遍历leafs无法到达，因此先加上这部分损失
			#2.加上arrived_leafs中所有结点及其兄弟结点的损失,这部分结点同样已经出队列，但未剪除，
			#因此对于loss_before和after_loss而言都要加回来,但要知道二者加的量是一致的，不影响大小
			#比较，因此可以不加
			loss_before = 0.0
			loss_before += self._loss(nd)
			#剪枝后的损失计算初始化：由于nd._parent尚未入队，遍历leafs无法到达，因此先加上这部分损失
			loss_after = 0.0
			loss_after = self._loss(nd._parent)
			'''被忽略(不影响大小比较)的增量
			for lf in arrived_leafs:
				for l in self._cur_model.children(lf._parent):
					loss_before += self._loss(l)
					loss_after += self._loss(l)
			'''
			'''
			遍历leafs累计剪枝前后所有叶结点的损失
			需要明确的一点是，由于存在new_leafs和arrived_leafs的过滤代码，此时leafs中的结点都可以
			如同刚进入循环一样等同处理
			'''
			for leaf in leafs:
				cur_loss = self._loss(leaf)
				if leaf._parent is not nd._parent:
					loss_after += cur_loss
				loss_before += cur_loss
			#追加对叶结点数量的惩罚项
			loss_before += alpha_leaf * len_before
			loss_after += alpha_leaf * len_after
			#决定是否剪枝
			if loss_after < loss_before:
				'''剪枝的处理：
				1.把父结点入队；
				2.父结点_label设置为所拥有样本的众数类；
				3.清空父结点所有孩子
				4.维护len_before变量为len_after，模型虽已经上提了叶结点，但是原来叶结点nd的
					兄弟结点还在队列中，遍历到这些兄弟结点时，由while循环开头利用new_leafs
					过滤掉即可
				'''
				leafs.append(nd._parent)
				if self._cur_model.is_root(nd._parent):
					_,ydata = self._get_examples(nd._parent)
					nd._parent._label = self._majority_class(ydata) 
				if self._cur_model.is_root(nd._parent):
					print('++++++++')
					nd._parent.showAttributes()
					print('++++++++')
				nd._parent._children.clear()
				len_before = len_after
					
			else:
				'''不剪枝的处理：
				1.将当前结点nd加入arrived_leafs中
				2.从new_leafs中删除本次while迭代尝试加入的nd._parent
				3.维护len_before变量，由于队列少了一个不符合剪枝条件的叶结点nd，直接减1即可，
					然后在while循环开头利用arrived_leafs进行过滤的代码中，每次都再将len_before
					减去1
				'''	
				arrived_leafs.add(nd)
				new_leafs.remove(nd._parent)
				len_before -= 1
				
			

			
			
			
			
	#---------------------------------公开方法--------------------------------
	def fit(self,xtrain=None,
			ytrain=None,
			examples=None,
			depth=None,
			max_depth=None,
			alpha_leaf=0,
			bool_prune=True):
		"""模型拟合的公开接口。若训练数据集未直接提供，则使用self._reader读取训练数据集
		Args:
			alpha_leaf:后剪枝对叶结点的正则化超参数,有效取值大于等于0.
		"""
		if xtrain is None or ytrain is None:
			self._fixdata()
			self._cur_model = self.DecisionTree()
			self._cur_model._root = self._fit(xtrain=self._reader._xtrain,
								ytrain=self._reader._ytrain,
								examples=range(len(self._reader._xtrain)),
								depth=-1)
		else:
			xtrain = np.asarray(xtrain)
			ytrain = np.asarray(ytrain)
			self._assert_xdata(xtrain)
			self._assert_ydata(ytrain)
			self._cur_model = self.DecisionTree()
			self._cur_model._root = self._fit(xtrain=xtrain,
								ytrain=ytrain,
								examples=range(len(self._reader._xtrain)),
								depth=-1)
		if bool_prune:
			self._prune(alpha_leaf=alpha_leaf)	
				
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
			cur_xtest = np.asarray(xtest)
			self._assert_xdata(cur_xtest)

		if use_model.is_leaf(use_model._root):
			preds = [use_model._root._label] * len(cur_xtest)
			return np.asarray(preds)

		preds = [None] * len(cur_xtest) 
		for i in range(len(cur_xtest)):
			node = use_model._root
			try:
				while node._label is None and node._children != {}:
					node = node._children[cur_xtest[i][node._feature]]
			except KeyError:
				raise self.PredictionError('待预测样本 {} 某个属性出现了新的取值!'.format(repr(cur_xtest[i])))
			if node._label is None:
				print('++++')
				node.showAttributes()
				print('++++')
				raise self.PredictionError('叶结点_label属性取值为None!')
			preds[i] = node._label	
		return np.asarray(preds)

	def eval(self,bool_use_stored_model=False,method=None):
		preds = self.predict(self._reader._xeval,bool_use_stored_model)	
		return preds,self._evaluator.eval(preds,self._reader._yeval,method) 

	def save_model(self,path=None):
		'''决策树分类器序列化'''
		if self.bool_not_trained():
			raise self.NotTrainedError('无法进行模型序列化，因为决策树分类器尚未训练!')
		if path is None:
			cur_path = self._reader._dataDir + '/ID3Classifier.pkl'
		else:
			cur_path = path
		import pickle
		with open(cur_path,'wb') as f:
			pickle.dump(self._cur_model,f)
		print('save_model done!')
	
	def load_model(self,path=None):
		'''载入模型'''
		if path is None:
			cur_path = self._reader._dataDir + '/ID3Classifier.pkl'
		else:
			cur_path = path
		import pickle
		with open(cur_path,'rb') as f:
			self._stored_model = pickle.load(f)
		print('load_model done!')




if __name__ == '__main__':
	obj = ID3Classifier(dataDir='/home/michael/data/GIT/MachineLearning/data/forID3')
	#print(obj._reader._xtrain)
	#obj._fixdata()
	#print(obj._reader._xtrain)
	obj.fit(alpha_leaf=0.01)
	#obj.save_model()
	#obj.load_model()
	#print('*************')
	#print(obj._cur_model)
	#print('*************')
	#obj._cur_model._root.showAttributes()
	#print(obj._stored_model)
	#验证集上预测结果
	print(obj.eval(bool_use_stored_model=False)[0])
	print(obj.eval(bool_use_stored_model=False)[1])
	#print('---')
	#for node in obj._cur_model.preorder():
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
