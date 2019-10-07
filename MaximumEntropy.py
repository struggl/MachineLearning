"""最大熵模型实现分类"""
import collections

from MLBase import ClassifierBase

def feat_1(x):
	return 1 if x[3] == 0 else 0

def feat_2(x):
	return 1 if x[3] == 1 else 0

def feat_3(x):
	return 1 if x[3] == 2 else 0

def feat_4(x):
	return 1 if x[1] == 0 else 0

def feat_5(x):
	return 1 if x[1] == 1 else 0

def feat_6(x):
	return 1 if x[1] == 2 else 0

class MaximumEntropy(ClassifierBase):
	def __init__(self,dataDir,reader=None):
		super().__init__(dataDir,reader)
		self._cur_model = self.get_feat_funs()
	
	def bool_not_trained(self):
		return False

	def save_model(self,path=None):
		'''最大熵分类器序列化'''
		if self.bool_not_trained():
			raise self.NotTrainedError('无法进行模型序列化，因为最大熵分类器尚未训练!')
		if path is None:
			cur_path = self._reader._dataDir + '/MaximumEntropy.pkl'
		else:
			cur_path = path
		import pickle
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
		import pickle
		with open(cur_path,'rb') as f:
			model = pickle.load(f)
		self._stored_model = self.get_feat_funs()
		for k,v in model.items():
			self._stored_model[k][1] = model[k]
		print('load_model done!')

	def get_feat_funs(self):
		'''返回特征函数字典，值为列表，列表的第一个元素为特征函数(全局定义)，第二个元素为权重'''
		feat_funs = collections.OrderedDict()
		feat_funs['feat_1'] = [feat_1,0]
		feat_funs['feat_2'] = [feat_2,0]
		feat_funs['feat_3'] = [feat_3,0]
		feat_funs['feat_4'] = [feat_4,0]
		feat_funs['feat_5'] = [feat_5,0]
		feat_funs['feat_6'] = [feat_6,0]
		
		return feat_funs

	def fit(self,xtrain,ytrain):
		pass

	def predict(self,xtest):
		pass

	def eval(self,predicts,labels,method=None):
		pass
	

if __name__ == '__main__':
	obj = MaximumEntropy(dataDir='/home/michael/data/GIT/MachineLearning/data/forMaximumEntropy')
	#obj.save_model()
	obj.load_model()
	print(obj._cur_model)
