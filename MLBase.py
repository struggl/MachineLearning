from abc import ABCMeta,abstractmethod
import numpy as np

class LearnerBase(metaclass=ABCMeta):
	"""所有学习器的基类"""
	__slots__ = '_learnerType','_evaluator','_reader'
	def __init__(self,learnerType=None,evaluator=None,reader=None):
		self._rearnerType = learnerType
		self._evaluator = evaluator
		self._reader = reader

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
		assert 'int' in preds.dtype and 'int' in labels.dtype
		assert len(preds) != 0 and len(preds) == len(labels)

		return sum(preds==labels) / len(preds)	
	
	def _getRecall(self,preds,labels):
		'''计算各类别的召回，返回一个字典'''
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
			lbs[lb] = num_right / num_lb
		return lbs	

	def _getPrecision(self,preds,labels):
		'''计算各类别的精度，返回一个字典'''
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
			lbs[lb] = num_right / (num_lb+self.smallDigit)
		return lbs	

	def _getF1score(self,preds,labels):
		'''计算各类别的f1-score，返回一个字典'''
		P = self._getPrecison(preds,labels)
		R = self._getRecall(preds,labels)
		F1 = dict()
		#注意，对于一个特定的类，P与R可能都取0
		for k in P.keys():
			F1[k] = 2*P[k]*R[k] / (P[k]+R[k]+self.smallDigit)	
		return F1		
			
	#----------------------------------公开接口----------------------------------
	def get_all_evaluation_method(self):
		'''获取所有支持的评价方法'''
		return {'accuracy','recall','f1-score'}
		
	def eval(self,predictions,labels,method=None):
		'''指定方法对模型结果进行评价
		Args:
			method:str.可选值在get_all_evaluation_method返回的集合中
		'''		
		if method is None:
			return self._getF1score(predictions,labels)
		
		if method not in self.get_all_evaluation_method():
			raise ValueError('method参数仅支持以下取值: '+repr(self.get_all_evaluation_method()))

class RegressorEvaluator(EvaluatorBase):
	"""回归模型的评价器"""
	def __init__(self):
		self._EvaluatorType = 'Regressor'

class ClassifierBase(LearnerBase):
	"""所有分类器的基类"""
	def __init__(self,data_path,reader=None):
		self._learnerType = 'Classifier'
		self._evaluator = ClassifierEvaluator()	
		self._reader = reader if reader is not None else tsvReader(data_path)

class RegressorBase(LearnerBase):
	"""所有回归器的基类"""
	def __init__(self):
		self._LearnerType = 'Classifier'
		self._Evaluator = RegressorEvaluator()	



class ReaderBase(metaclass=ABCMeta):
	"""数据读取器基类,读取器对象存储数据，并提供数据预处理的方法"""
	__slots__ = '_dataPath'
	
	def __init__(self,dataPath=None):
		self._dataPath = dataPath
	
	@abstractmethod
	def read(self):
		pass

	
class tsvReader(ReaderBase):
	'''tsv格式的读取器'''
	def __init__(self,dataPath):
		super().__init__(dataPath)

	def read(self):	

class NotTrainedError(Exception):
	pass	

class DecisionTreeClassifierBase(ClassifierBase):
	'''决策树分类器的基类'''	
	__slots__ = '_tree'	

	def __init__(self,data_path,reader=None):
		super().__init__(data_path,reader)
		self._tree = None

	def _fixdata(self,xtrain):
		raise NotImplementedError	

	def _majority_class(self,ytrain):
		freq = {}
		for lb in ytrain:
			freq[lb] += 1	
		return sorted(freq.items(),key=lambda x:x[1],reverse=True)[0][0]

	def _chooseBestFeatureToSplit(self,xtrain,ytrain):
		'''选择最优划分特征'''
		raise NotImplementedError

	def _splitDataSet(self,xtrain,ytrain,feat,val):
		'''根据特征的特定值获取子数据集，该数据集不再含有指定的feat列'''
		rows = set()
		for i in range(len(xtrain)):
			if feat[i][feat] == val:
				rows.add(i)	
		xtrain = [None]*len(rows)
		ytrain = [None]*len(rows)
		for rw in rows:
			xtrain[rw] = xtrain[rw][:feat].extend(xtrain[rw][feat+1:])
			ytrain[rw] = ytrain[rw]
		return np.asarray(xtrain),np.asarray(ytrain)
	
	def _fit(self,xtrain,ytrain):
		'''训练决策树分类器'''
		assert type(xtrain) == np.ndarray and type(ytrain) == np.ndarray
		assert xtrain.ndim == 2	and ytrain.ndim == 1
		assert 'int' in ytrain.dtype				

		self._fixdata()			#修复数据的钩子，例如ID3决策树要求所有数值均为标称型数据	

		if ytrain.count(ytrain[0]) == len(ytrain):	#递归返回情况1：所有类别相同
			return tuple(ytrain[0])

		if len(xtrain[0]) == 1:				#递归返回情况2：无法继续切分特征时，返回众数类
			return tuple(_majority_class(ytrain))
		
		bestFeat = self._chooseBestFeatureToSplit(xtrain,ytrain)	#选择最优划分特征
		bestFeatVals = set([example[bestFeat] for example in xtrain])
		
		resDict = {bestFeat:{}}
		for val in bestFeatVals:	#对最优特征的每个值尝试构建子树	
			cur_xtrain,cur_ytrain = self._splitDataSet(xtrain,ytrain,bestFeat,val)
			resDict[val] = self._fit(cur_xtrain,cur_ytrain)
		return resDict	
			
	'''---------------------------------公开方法--------------------------------'''
	def fit(self):
		"""模型拟合的公开接口，根结点统一使用'root'作为键，包括根结点在内的所有内部结点都是dict对象，唯独叶结点是tuple对象
		模型示例一(当数据集仅有一个特征而导致决策树直接预测众数类或者数据集所有类别相同时)：
		#注意，此处label1的取值可能与特征列索引数字重合
		{
		'root':
			(label1,)
		}

		模型的形式二：
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
		"""
		self._reader.read()	#使用数据读取器对象读取数据
		self._tree = {}
		self._tree['root'] = self._fit(self._xtrain,self._ytrain)
				
	def prediect(self,xtest):
		'''模型预测的公开接口'''
		#assert type(xtest) is np.ndarray
		#assert xtest.ndim == 2 
		if self._tree is None or self._tree['root'] == {}:
			raise NotTrainedError('决策树分类器尚未训练!')
		self._fixdata()	

		preds = [None] * len(xtest) 
		for i in range(len(xtest)):
			tree = self._tree['root']
			#仅当tree变量是字典的时候才可以迭代feat,否则可能引发IndexError,下面while的处理也是这个原因
			if type(tree) is dict:		
				feat = next(iter(tree.keys()))
			while isinstance(tree,dict):
				tree = tree[feat]
				if type(tree) is dict:
					feat = xtest[i][feat]
			preds[i] = tree[0]
		return np.asarray(preds)

if __name__ == '__main__':
	obj = ClassifierEvaluator()
	obj.eval('a','b','c')

