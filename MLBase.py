from abc import ABCMeta,abstractmethod
import numpy as np

class LearnerBase(metaclass=ABCMeta):
	"""所有学习器的基类"""
	@abstractmethod
	def fit(self,xtrain,ytrain):
		pass

	@abstractmethod:
	def predict(self,xtest):
		pass
	
	@abstractmethod:
	def eval(self,predicts,labels,method=None):
		pass

	@abstractmethod:
	def load_model(self,path):
		pass

	@abstractmethod:
	def save_model(self,path):
		pass
	
	def getLearnerType(self):
		return self._LearnerType


class EvaluatorBase(metaclass=ABCMeta):
	"""所有模型评价指标的基类"""
	@abstractmethod
	def get_all_evaluation_method(self):
		'''获取所有支持的评价方法'''
		pass	
	
	@abstractmethod
	def eval(predicts,labels,method=None):
		'''根据模型预测与数据指标进行模型评测
		Args:
			predicts:模型预测结果
			labels:数据原来的标签
			method:str.指定使用的评价方法
		'''
	def getEvaluatorType(self):
		return self._EvaluatorType

class ClassifierEvaluator(EvaluatorBase):
	"""分类模型的评价器"""
	def __init__(self):
		self._EvaluatorType = 'Classifier'
		
class RegressorEvaluator(EvaluatorBase):
	"""回归模型的评价器"""
	def __init__(self):
		self._EvaluatorType = 'Regressor'

class ClassifierBase(LearnerBase):
	"""所有分类器的基类"""
	def __init__(self):
		self._LearnerType = 'Classifier'
		self._Evaluator = ClassifierEvaluator()	

class RegressorBase(LearnerBase):
	"""所有回归器的基类"""
	def __init__(self):
		self._LearnerType = 'Classifier'
		self._Evaluator = RegressorEvaluator()	



class ReaderBase(metaclass=ABCMeta):
	"""数据读取器基类"""
	def __init__(self,data_path):
		self.data_path = data_path	

	@abstractmethod
	def read(self):
		pass

	
class tsvReader(ReaderBase):
	def __init__(self):
		super().__init__()		

	def read(self):
		pass

class NotTrainedError(Exception):
	pass	

class DecisionTreeClassifierBase(ClassifierBase):
	'''决策树分类器的基类'''	
	def __init__(self):
		super().__init__()
		self._xtrain = None
		self._ytrain = None
		self._tree = None

	def _fixdata(xtrain):
		raise NotImplementedError	

	def _majority_class(ytrain):
		freq = {}
		for lb in ytrain:
			freq[lb] += 1	
		return sorted(freq.items(),key=lambda x:x[1],reverse=True)[0][0]

	def _chooseBestFeatureToSplit(xtrain,ytrain):
		'''选择最优划分特征'''
		raise NotImplementedError

	def _splitDataSet(xtrain,ytrain,feat,val):
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
		assert type(xtrain) is np.ndarray and type(ytrain) is np.ndarray
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
	def reset_tree(self):
		'''重置决策树分类器'''
		self._tree = {'root':None}

	def fit(self,xtrain,ytrain):
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
		self._tree = {}
		self._tree['root'] = self._fit(self._xtrain,self._ytrain)
				
	def prediect(xtest):
		'''模型预测的公开接口'''
		assert type(xtest) is np.ndarray
		assert xtest.ndim == 2 
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
