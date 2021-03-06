from abc import ABCMeta,abstractmethod
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
	def __init__(self,smallDigit=None):
		super().__init__(smallDigit)
		self._EvaluatorType = 'Regressor'

	def _getMSE(self,predicts,ydata):
		'''计算模型预测值与真实值的均方误差MSE'''
		predicts = np.asarray(predicts,dtype='float64')
		ydata = np.asarray(ydata,dtype='float64')
		n = len(ydata)
		if n == 0:
			raise ValueError('模型预测值predicts和真实值ydata的长度不能为0')
		
		square_error = sum( (ydata - predicts) ** 2 )
		n = len(ydata)
		if n == 1:
			return square_error
		mse = square_error / (n-1)
		return mse

	def get_all_evaluation_method(self):
		'''获取所有支持的评价方法'''
		return {'mse'}	
	
	def eval(self,predicts,labels,method=None):
		'''指定方法对模型结果进行评价
		Args:
			method:str.可选值在get_all_evaluation_method返回的集合中
		'''		
		if method is None or method == 'mse':
			return self._getMSE(predicts,labels)	
	
		if method not in self.get_all_evaluation_method():
			raise ValueError('method参数仅支持以下取值: '+repr(self.get_all_evaluation_method()))


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
	__slots__ = '_learnerType','_evaluator','_reader','_cur_model','_stored_model'
	def __init__(self,dataDir,reader=None):
		self._learnerType = 'Regressor'
		self._evaluator = RegressorEvaluator()	
		self._reader = reader if reader is not None else tsvReader(dataDir)
		self._reader.read()
		self._cur_model = None
		self.stored_model = None


class LabelerBase(LearnerBase):
	'''序列标注器的基类'''
	__slots__ = '_learnerType','_evaluator','_reader','_cur_model','_stored_model'
	def __init__(self,dataDir,reader=None):
		self._learnerType = 'Labeler'
		self._evaluator = LabelerEvaluator()	
		self._reader = reader if reader is not None else tsvReaderForLabeler(dataDir)
		self._reader.read()
		self._cur_model = None
		self.stored_model = None
		


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
			
class DecisionTreeRegressorBase(RegressorBase):
	'''决策树回归器基类'''
	def bool_not_trained(self,tree=None):
		'''判断决策树是否已经训练'''
		raise NotImplementedError

	def _assert_xdata(self,xdata):
		xdata = np.asarray(xdata)
		assert xdata.ndim == 2 
		assert len(xdata) != 0
		return xdata

	def _assert_ydata(self,ydata):
		ydata = np.asarray(ydata)
		assert ydata.ndim == 1
		assert ydata.dtype == 'float64' or ydata.dtype == 'float32'
		assert len(ydata) != 0
		return ydata

	def _calSSE(self,ydata):
		'''计算平方误差和'''
		ydata = np.asarray(ydata,dtype='float64')
		n = len(ydata)
		if n == 0:
			return 0.0
		SSE = sum( (ydata - np.mean(ydata)) ** 2 )
		return SSE

	def _splitDataSet(self,xtrain,ytrain,feat,val):
		'''获取子数据集feat列上取值为val的子数据集
		Args:
			xtrain:np.ndarray,二维特征向量
			ytrain:np.ndarray,维度是一。
			feat:python int.指定列索引
			val:python int.指定feat列的取值
		'''
		raise NotImplementedError


	class Node(object):
		'''决策树的结点类'''
		def __init__(self):
			raise NotImplementedError
	
		def showAttributes(self):
			raise NotImplementedError

	
	class DecisionTree(object):
		'''决策树数据结构'''
		def __init__(self):
			self._size = 0
			self._root = None

		def __len__(self):
			return self._size

		def _validate(self,node):
			raise NotImplementedError

		def is_leaf(self,node):
			raise NotImplementedError

		def is_root(self,node):
			raise NotImplementedError
		#-----------------------------访问方法-----------------------------
		def preOrder(self,node=None):
			'''从node开始进行前序遍历，若node为None，则从根开始遍历,返回一个迭代器'''
			raise NotImplementedError

		def parent(self,node):
			'''返回给定node的父结点'''
			raise NotImplementedError

		def children(self,node):
			'''返回给定结点node的孩子结点的迭代器'''
			raise NotImplementedError

		def sibling(self,node):
			'''返回给定node结点的兄弟结点'''
			raise NotImplementedError

		def num_children(self,node):
			'''返回给定node结点的非node孩子数'''
			raise NotImplementedError
			
		
		#-----------------------------更新方法------------------------------				
		def add_root(self,node):
			'''为决策树添加根结点，根结点深度设定为1'''
			raise NotImplementedError

class DecisionTreeClassifierBase(ClassifierBase):
	'''决策树分类器基类'''
	def bool_not_trained(self,tree=None):
		'''判断决策树是否已经训练'''
		raise NotImplementedError
		
	def _assert_xdata(self,xdata):
		xdata = np.asarray(xdata)
		assert xdata.ndim == 2 
		assert len(xdata) != 0
		return xdata

	def _assert_ydata(self,ydata):
		ydata = np.asarray(ydata)
		assert ydata.ndim == 1
		assert ydata.dtype == 'int64' or ydata.dtype == 'int32'
		assert len(ydata) != 0
		return ydata

	def _majority_class(self,ytrain):
		ytrain = self._assert_ydata(ytrain)
		freq = {}
		for lb in ytrain:
			freq[lb] = freq.get(lb,0) + 1	
		return sorted(freq.items(),key=lambda x:x[1],reverse=True)[0][0]

	def _calInformationEntropy(self,ydata):
		'''给定数据集D，计算数据集D的信息熵,信息熵取值越小越好.令K为类别数量，
		Ent(D) = - sum_{k=1}^K p_k*log(p_k,2),其中p_k为第k类在数据集中出现的频率
		Ent(D)最小值为0，最大值为log(K,2)
		'''
		ydata = self._assert_ydata(ydata)
		from math import log
		totalEnt = 0.0
		lbFreq = {}
		for lb in ydata:
			lbFreq[lb] = lbFreq.get(lb,0) + 1
		nexample = len(ydata)	
		for k in lbFreq.keys():
			p_k = float(lbFreq[k]) / nexample
			totalEnt -= p_k * log(p_k,2) 
		return totalEnt
			
	def _calGini(self,ydata):
		'''给定数据集D，计算数据集D的基尼指数，基尼指数取值越小越好。令K为类别数量，
		Gini(D) = 1 - sum_{k=1}^K p_k^2,其中p_k为第k类在数据集中出现的频率
		'''	
		ydata = self._assert_ydata(ydata)
		lbFreq = {}
		for lb in ydata:
			lbFreq[lb] = lbFreq.get(lb,0) + 1
		nexample = len(ydata)	
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
		xdata = self._assert_xdata(xdata)
		ydata = self._assert_ydata(ydata)
		baseInformationEntropy = self._calInformationEntropy(ydata)
		featValSet = set([example[feat] for example in xdata])
		nexample = float(len(xdata))
		newInformationEntropy = 0.0
		for val in featValSet:
			_,cur_ydata = self._splitDataSet(xdata,ydata,feat,val)
			p_v = len(cur_ydata) / nexample
			newInformationEntropy += p_v * self._calInformationEntropy(cur_ydata)
		InformationGain = baseInformationEntropy - newInformationEntropy
		return InformationGain	
												
	def _calInformationGainRatio(self,xdata,ydata,feat):
		'''给定数据集D和属性feat，计算属性feat的信息增益率'''
		xdata = self._assert_xdata(xdata)
		ydata = self._assert_ydata(ydata)
		entropy_feat = 0.0
		featValSet = dict()
		
		feat_column = [example[feat] for example in xdata]
		for val in feat_column:
			featValSet[val] = featValSet.get(val,0) + 1 
		nexample = float(len(xdata))

		from math import log
		for v in featValSet.values():
			p_v = v / nexample
			entropy_feat -= p_v * log(p_v,2)
		
		informationGain = self._calInformationGain(xdata,ydata,feat)
		return float(informationGain) / entropy_feat
			
		

	def _calGiniIndex(self,xdata,ydata,feat):
		'''给定数据集D和属性feat，计算属性feat的基尼指数'''
		xdata = self._assert_xdata(xdata)
		ydata = self._assert_ydata(ydata)
			
	def _splitDataSet(self,xtrain,ytrain,feat,val):
		'''获取子数据集feat列上取值为val的子数据集
		Args:
			xtrain:np.ndarray,二维特征向量
			ytrain:np.ndarray,维度是一。
			feat:python int.指定列索引
			val:python int.指定feat列的取值
		'''
		raise NotImplementedError


	class Node(object):
		'''决策树的结点类'''
		def __init__(self):
			raise NotImplementedError
	
		def showAttributes(self):
			raise NotImplementedError

	
	class DecisionTree(object):
		'''决策树数据结构'''
		def __init__(self):
			self._size = 0
			self._root = None

		def __len__(self):
			return self._size

		def _validate(self,node):
			raise NotImplementedError

		def is_leaf(self,node):
			raise NotImplementedError

		def is_root(self,node):
			raise NotImplementedError
		#-----------------------------访问方法-----------------------------
		def preOrder(self,node=None):
			'''从node开始进行前序遍历，若node为None，则从根开始遍历,返回一个迭代器'''
			raise NotImplementedError

		def parent(self,node):
			'''返回给定node的父结点'''
			raise NotImplementedError

		def children(self,node):
			'''返回给定结点node的孩子结点的迭代器'''
			raise NotImplementedError

		def sibling(self,node):
			'''返回给定node结点的兄弟结点'''
			raise NotImplementedError

		def num_children(self,node):
			'''返回给定node结点的非node孩子数'''
			raise NotImplementedError
			
		
		#-----------------------------更新方法------------------------------				
		def add_root(self,node):
			'''为决策树添加根结点，根结点深度设定为1'''
			raise NotImplementedError

