from abc import ABCMeta,abstractmethod
import numpy as np

class LearnerBase(metaclass=ABCMeta):
	"""所有学习器的基类"""
	@abstractmethod
	def fit(self,xtrain,ytrain):
		pass

	@abstractmethod
	def predict(self,xtest):
		pass
	
	@abstractmethod
	def eval(self,predicts,labels,method=None):
		pass

	#@abstractmethod
	def load_model(self,path):
		pass

	#@abstractmethod
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

	def _read(self,path,bool_read_first_line=True):
		'''读取数据为np.ndarray,且dtype为float64,默认最后一列为标签列'''
		fr = open(path,'r',encoding='utf-8')
		x = []
		y = []
		n = 0
		for line in fr:
			if bool_read_first_line == False and n == 0:
				continue
			example = [float(feat) for feat in line.strip().split('\t')]
			x.append(example[:-1])
			y.append(example[-1])
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
			

class NotTrainedError(Exception):
	pass	


class DecisionTreeClassifierBase(ClassifierBase):
	'''决策树分类器基类'''
	def _majority_class(self,ytrain):
		freq = {}
		for lb in ytrain:
			freq[lb] = freq.setdefault(lb,0) + 1	
		return sorted(freq.items(),key=lambda x:x[1],reverse=True)[0][0]

	def _assert_xdata(self,xdata):
		assert type(xdata) is np.ndarray
		assert xdata.ndim == 2 
		assert len(xdata) != 0

	def _assert_ydata(self,ydata):
		assert type(ydata) is np.ndarray
		assert ydata.ndim == 1
		assert ydata.dtype == 'int64' or ydata.dtype == 'int32'
		assert len(ydata) != 0

	def _calInformationEntropy(self,xdata,ydata):
		'''给定数据集D，计算数据集D的信息熵,信息熵取值越小越好.令K为类别数量，
		Ent(D) = - sum_{k=1}^K p_k*log(p_k,2),其中p_k为第k类在数据集中出现的频率
		'''
		from math import log
		totalEnt = 0.0
		lbFreq = {}
		for lb in ydata:
			lbFreq[lb] = lbFreq.setdefault(lb,0) + 1
		nexample = len(xdata)	
		for k in lbFreq.keys():
			p_k = float(lbFreq[k]) / nexample
			totalEnt -= p_k * log(p_k,2) 
		return totalEnt
			
	def _calGini(self,xdata,ydata):
		'''给定数据集D，计算数据集D的基尼指数，基尼指数取值越小越好。令K为类别数量，
		Gini(D) = 1 - sum_{k=1}^K p_k^2,其中p_k为第k类在数据集中出现的频率
		'''	
		lbFreq = {}
		for lb in ydata:
			lbFreq[lb] = lbFreq.setdefault(lb,0) + 1
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
		pass

	def _calGiniIndex(self,xdata,ydata,feat):
		'''给定数据集D和属性feat，计算属性feat的基尼指数'''
		self._assert_xdata(xdata)
		self._assert_ydata(ydata)
			
	def _splitDataSet(self,xtrain,ytrain,feat,val,bool_contain_feat_column=False):
		'''获取子数据集feat列上取值为val的子数据集
		Args:
			xtrain:np.ndarray,二维特征向量
			ytrain:np.ndarray,维度是一。
			feat:python int.指定列索引
			val:python int.指定feat列的取值
			bool_contain_feat_column:bool.若为True,则采用CART决策树的二元划分策略，子数据集仍然包含feat这一列;
				否则子数据集不保留feat所在列。
		'''
		if bool_contain_feat_column:
			raise NotImplementedError
		xdata = []
		ydata = []
		for i in range(len(xtrain)):
			if xtrain[i][feat] == val:
				xdata.append(np.concatenate([xtrain[i][:feat],xtrain[i][feat+1:]]))
				ydata.append(ytrain[i])
		return np.asarray(xdata,dtype=xtrain.dtype),np.asarray(ydata,dtype=ytrain.dtype)

			
	

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

	def _fit(self,xtrain,ytrain):
		'''训练决策树分类器'''
		freq = 0
		for lb in ytrain:
			if lb == ytrain[0]:
				freq += 1
		if freq == len(ytrain):	#递归返回情况1：所有类别相同
			return (ytrain[0],)

		bestFeat = self._chooseBestFeatureToSplit(xtrain,ytrain)	#选择最优划分特征
		if bestFeat is None:				#递归返回情况2：无法继续切分特征时，返回众数类
			return (self._majority_class(ytrain),)
		
		bestFeatVals = set([example[bestFeat] for example in xtrain])
		
		resDict = {bestFeat:{}}
		for val in bestFeatVals:	#对最优特征的每个值构建子树	
			cur_xtrain,cur_ytrain = self._splitDataSet(xtrain,ytrain,bestFeat,val)
			resDict[bestFeat][val] = self._fit(cur_xtrain,cur_ytrain)
		return resDict	

	def _fixdata(self):
		self._reader._xtrain = np.asarray(self._reader._xtrain,dtype='int64')
		self._reader._xtest = np.asarray(self._reader._xtest,dtype='int64')
		self._reader._xeval = np.asarray(self._reader._xeval,dtype='int64')
			
	'''---------------------------------公开方法--------------------------------'''
	def fit(self,xtrain=None,ytrain=None):
		"""模型拟合的公开接口。若训练数据集未直接提供，则使用self._reader读取训练数据集
		决策树模型中，根结点统一使用'root'作为键，包括根结点在内的所有内部结点都是dict对象，唯独叶结点是tuple对象
		模型示例一(当数据集仅有一个特征而导致决策树直接预测众数类或者数据集所有类别相同时)：
		#注意，此处label1的取值可能与特征列索引数字重合
		{
		'root':
			(label1,)
		}

		模型的形式二：
		特点是，进入root结点后，若为字典，则仅有一个键，该键对应着特征的序号，
		接着是一个可能拥有多个键的字典，不同的键对应着由于特征向量上该特征的不同取值而划分的不同子树
		{
		'root':
			{
			0:
				{
				0:('100',)
				1:
					{
					0:('100',)
					1:('10000',)
					}
				}
			}
		}
		###实际上建立的树为：
		'root': {
			3: 
				{
				0: 
					{
					1: 
						{
						0: (1,), 
						1: 
							{
							0: 
								{
								0: (1,), 
								1: 
									{
									2: 
										{
										0: (1,), 
										1: (0,)
										}
									}
								}
							}, 
						2: (0,)
						}
					}, 
				1: (0,), 
				2: (0,)
				}
			}
		}
		"""
		self._cur_model = {}
		if xtrain is None or ytrain is None:
			self._fixdata()
			self._cur_model['root'] = self._fit(self._reader._xtrain,self._reader._ytrain)
		else:
			self._assert_xdata(xtrain)
			self._assert_ydata(ytrain)
			self._cur_model['root'] = self._fit(xtrain,ytrain)
			
				
	def predict(self,xtest=None,bool_use_stored_model=False):
		'''模型预测的公开接口'''
		if bool_use_stored_model:
			use_model = self._stored_model
		else:
			use_model = self._cur_model
		if use_model is None or use_model['root'] == {}:
			raise NotTrainedError('决策树分类器尚未训练!')

		if xtest is None:
			cur_xtest = self._reader.xtest
		else:
			cur_xtest = np.asarray(xtest)
			self._assert_xdata(cur_xtest)

		preds = [None] * len(cur_xtest) 
		for i in range(len(cur_xtest)):
			tree = use_model['root']
			#仅当tree变量是字典的时候才可以迭代feat,否则可能引发IndexError,下面while的处理也是这个原因
			if type(tree) is dict:		
				#当前分裂点对应的特征
				feat = next(iter(tree.keys()))
			while isinstance(tree,dict):
				#根据当前分裂点对应特征的取值进入子树
				feat_val = cur_xtest[i][feat]
				tree = tree[feat][feat_val]
				if type(tree) is dict:
					feat = next(iter(tree.keys()))
			if not isinstance(tree,dict):
				preds[i] = tree[0]
		return np.asarray(preds)

	def eval(self,method=None):
		preds = self.predict(self._reader._xeval,bool_use_stored_model=False)	
		return preds,self._evaluator.eval(preds,self._reader._yeval) 
			

if __name__ == '__main__':
	obj = ID3Classifier(dataDir='/home/michael/data/GIT/MachineLearning/data')
	#print(type(obj._reader))
	print(obj._reader._xtrain)
	#print(obj._reader._xtrain.dtype)
	obj._fixdata()
	print(obj._reader._xtrain)
	#print(obj._reader._xtrain.dtype)
	obj.fit()
	print(obj._cur_model)
	#print 验证集上预测结果
	print(obj.eval()[0])
	#print 验证集上评价结果
	print(obj.eval()[1])
	#执行预测
	print(obj.predict([[0,0,0,0,0,0]]))
	print(obj.predict([[1,1,1,1,1,0]]))
