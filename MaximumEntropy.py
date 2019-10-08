"""最大熵模型实现分类"""
import collections
import math
import numpy as np
import pickle

from MLBase import ClassifierBase

def feat_1(x,y):
	return 1 if x[3] == 0 and y == 0 else 0

def feat_2(x,y):
	return 1 if x[3] == 1 and y == 1 else 0

def feat_3(x,y):
	return 1 if x[3] == 2 and y == 1 else 0

def feat_4(x,y):
	return 1 if x[1] == 0 else 0

def feat_5(x,y):
	return 1 if x[1] == 1 else 0

def feat_6(x,y):
	return 1 if x[1] == 2 else 0

class MaximumEntropy(ClassifierBase):
	def __init__(self,dataDir,reader=None):
		super().__init__(dataDir,reader)
		self._cur_model = self.get_feat_funs()
	
	def get_feat_funs(self):
		'''返回特征函数字典，值为列表，列表的第一个元素为特征函数(全局定义)，第二个元素为权重'''
		feat_funs = collections.OrderedDict()
		feat_funs['feat_1'] = [feat_1,0]
		'''
		feat_funs['feat_1'] = [feat_1,10]
		feat_funs['feat_2'] = [feat_2,2]
		feat_funs['feat_3'] = [feat_3,3]
		feat_funs['feat_4'] = [feat_4,4]
		feat_funs['feat_5'] = [feat_5,0]
		feat_funs['feat_6'] = [feat_6,0]
		'''
		return feat_funs
			
	def bool_not_trained(self,use_model=None):
		if use_model is None:
			use_model = self._cur_model
		if use_model is None:
			return True
		return False

	def save_model(self,path=None):
		'''最大熵分类器序列化'''
		if self.bool_not_trained():
			raise self.NotTrainedError('无法进行模型序列化，因为最大熵分类器尚未训练!')
		if path is None:
			cur_path = self._reader._dataDir + '/MaximumEntropy.pkl'
		else:
			cur_path = path
		model = dict()
		for k,v in self._cur_model.items():
			model[k] = v[1]
		with open(cur_path,'wb') as f:
			pickle.dump(model,f)
		print('save_model done!')
	
	def load_model(self,path=None):
		'''载入模型'''
		if path is None:
			cur_path = self._reader._dataDir + '/MaximumEntropy.pkl'
		else:
			cur_path = path
		with open(cur_path,'rb') as f:
			model = pickle.load(f)
		self._stored_model = self.get_feat_funs()
		for k,v in model.items():
			self._stored_model[k][1] = model[k]
		print('load_model done!')

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

	def _count_prob_x(self,xdata=None):
		'''计算特征向量的经验分布'''
		if xdata is None:
			xdata = self._reader._xtrain
		xdata = self._assert_xdata(xdata)
		Dict = dict()
		for x in xdata:
			Dict[tuple(x)] = Dict.get(tuple(x),0) + 1
		for k in Dict.keys():
			Dict[k] = Dict[k] / len(xdata)
		return Dict
	
	def _count_prob_xy(self,xdata=None,ydata=None):
		if xdata is None:
			xdata = self._reader._xtrain
		if ydata is None:
			ydata = self._reader._ytrain
		xdata = self._assert_xdata(xdata)
		ydata = self._assert_ydata(ydata)
		#将特征向量与标签列按列拼接
		data = np.c_[xdata,ydata]
		return self._count_prob_x(xdata=data)

	def test(self):
		print(self._count_prob_x())
		print('---')
		print(self._count_prob_xy())

	def fit(self,xtrain=None,ytrain=None,method=None,max_iterations=10):
		'''最大熵分类器训练的公开接口，默认使用IIS算法'''
		if xtrain is None or ytrain is None:
			xtrain = self._reader._xtrain
			ytrain = self._reader._ytrain
		if method is None or method == 'IIS':
			self._fit_IIS(xtrain,ytrain,max_iterations)
		elif method == 'BFGS':
			self._fit_BFGS(xtrain,ytrain)

	def _fit_BFGS(self,xtrain,ytrain):
		pass

	def _fit_IIS(self,xtrain,ytrain,max_iterations):
		'''根据李航<<统计学习方法>>第六章 使用改进的迭代尺度算法(IIS)实现最大熵分类器训练'''	
		if type(max_iterations) != int or max_iterations <= 0:
			raise ValueError('使用改进的迭代尺度算法训练最大熵分类器时，max_iterations参数必须为正整数!')
		prob_x = self._count_prob_x()
		prob_xy = self._count_prob_xy()
		#每轮迭代
		for i in range(max_iterations):
			#每个特征函数
			for feat in self._cur_model.keys():
				#计算p90式(6.32) 的第一项
				g_delta1 = 0.0
				for xy,prob in prob_xy.items():
					x = xy[:-1]
					y = xy[-1]
					if len(x) == 1:
						x = np.asarray([x])
					else:
						x = np.asarray(x)
					y = np.asarray([y])
					g_delta1 += prob * self._cur_model[feat][0](x,y)
				#计算p90式(6.32) 的第二项 和 p91式(6.35)的分母
				g_delta2 = 0.0
				g_delta_derivative = 0.0
				for x,prob in prob_x.items():
					x = np.asarray(x)
					#当前模型对x的预测结果为一个字典，每个项格式为(标签，预测概率)
					predDict = self._predict(x,use_model=self._cur_model)[1]	
					#记式(6.32)第二项对y求和项为Sum
					Sum = 0
					#计算g_delta_derivative表达式中对y的求和项
					Sum_derivative = 0.0
					#对预测结果字典的每个标签
					for lb in predDict.keys():
						y = [lb]
						#计算f^#(x,y)
						f_sharp_xy = sum( [val[0](x,y) for val in self._cur_model.values()] )
						tmp = predDict[lb] * self._cur_model[feat][0](x,y) *\
							math.exp( self._cur_model[feat][1] * f_sharp_xy )
						Sum += tmp
						Sum_derivative += tmp * f_sharp_xy
					g_delta2 += prob * Sum
					g_delta_derivative += prob * Sum_derivative
				g_delta = g_delta1 - g_delta2	
				#本次迭代，特征函数feat对应权重的减少量
				delta = g_delta / ( g_delta_derivative + self._evaluator._smallDigit )
				self._cur_model[feat][1] -= delta
						
	def _predict(self,xtest,use_model=None):
		'''指定模型对单个向量进行预测，返回预测值与各个类别对应的概率'''
		if use_model is None:
			raise TypeError('参数use_model不能为None!')
		probs = dict()	
		for lb in set(self._reader._ytrain):
			prob = 0.0
			for k in use_model.keys():
				prob += use_model[k][0](xtest,lb) * use_model[k][1]
			prob = math.exp(prob)
			probs[lb] = prob
		Sum = sum(probs.values()) + self._evaluator._smallDigit
		for k in probs.keys():
			probs[k] = probs[k] / Sum 
		pred = sorted(probs.items(),key=lambda x:x[1],reverse=True)[0][0]
		return pred,probs
		
	def predict(self,xtest=None,bool_use_stored_model=False):
		'''模型预测的公开接口'''
		if bool_use_stored_model:
			use_model = self._stored_model
		else:
			use_model = self._cur_model
		if self.bool_not_trained(use_model):
			raise self.NotTrainedError('无法进行模型预测，因为最大熵分类器尚未训练!')
		if xtest is None:
			cur_xtest = self._reader._xtest		
		else:
			cur_xtest = self._assert_xdata(xtest)

		preds = [None] * len(cur_xtest)
		for i in range(len(cur_xtest)):
			preds[i] = self._predict(xtest=cur_xtest[i],use_model=use_model)[0]
		preds = np.asarray(preds)
		return preds

	def eval(self,bool_use_stored_model=False,method=None):
		preds = self.predict(self._reader._xeval,bool_use_stored_model)	
		return preds,self._evaluator.eval(preds,self._reader._yeval,method) 
	

if __name__ == '__main__':
	obj = MaximumEntropy(dataDir='/home/michael/data/GIT/MachineLearning/data/forMaximumEntropy')
	print(obj._cur_model)
	print(obj.eval(bool_use_stored_model=False)[0])
	print(obj.eval(bool_use_stored_model=False)[1])
	print('----')
	obj.fit()
	print(obj._cur_model)
	print(obj.eval(bool_use_stored_model=False)[0])
	print(obj.eval(bool_use_stored_model=False)[1])
	
	#obj.load_model()
	#print(obj._stored_model)
	#print(obj.eval(bool_use_stored_model=True)[0])
	#print(obj.eval(bool_use_stored_model=True)[1])
	#print(obj.test())

