'''ID3决策树实现'''
import numpy as np
import collections

from MLBase import DecisionTreeClassifierBase

class ID3Classifier(DecisionTreeClassifierBase):
	'''ID3决策树分类器'''	
	def __init__(self,dataDir,reader=None):
		super().__init__(dataDir,reader)

	def bool_not_trained(self,tree=None):
		'''判断决策树是否已经训练,仅判断根结点，默认在_fit和_prune方法的更新过程中其余结点维护了相应的特性'''
		if tree is None:
			tree = self._cur_model
		if tree is None or tree._root is None:
			return True
		if tree._root._children == {} and tree._root._label is None:
			return True
		return False	

	def _splitDataSet(self,xtrain,ytrain,feat,val):
		'''获取子数据集feat列上取值为val的子数据集
		Args:
			xtrain:np.ndarray,二维特征向量
			ytrain:np.ndarray,维度是一。
			feat:python int.指定列索引
			val:python int.指定feat列的取值
		'''
		xtrain = self._assert_xdata(xtrain)
		ytrain = self._assert_ydata(ytrain)
		xdata = []
		ydata = []
		for i in range(len(xtrain)):
			if xtrain[i][feat] == val:
				xdata.append(np.concatenate([xtrain[i][:feat],xtrain[i][feat+1:]]))
				ydata.append(ytrain[i])
		xdata,ydata = np.asarray(xdata,dtype=xtrain.dtype),np.asarray(ydata,dtype=ytrain.dtype)
		return xdata,ydata
	
	def _chooseBestFeatureToSplit(self,xtrain,ytrain,epsion=0):
		'''使用信息熵选择最优划分特征,若数据集特征数不大于0或最优划分的信息增益大于阈值epsion，则返回None
		Args:
			epsion:每次结点划分时损失函数下降的阈值，默认为0
		'''
		numFeat = len(xtrain[0])
		if numFeat < 1:
			return None
		bestGain = epsion
		bestFeat = None
		for feat in range(numFeat):
			curGain = self._calInformationGain(xtrain,ytrain,feat)
			if curGain > bestGain:
				bestGain = curGain
				bestFeat = feat
		if bestFeat != None and bestGain > epsion:
			return bestFeat,bestGain

	def _fit(self,xtrain,ytrain,examples,depth,global_labels,max_depth):
		'''训练决策树分类器
		global_labels:python list.指定xtrain每一列对应的标签名称，一般而言使用list(range(len(xtrain[0])))即可
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
		
		#选择最优划分特征
		res = self._chooseBestFeatureToSplit(xtrain,ytrain)
		if res is None:	#递归返回情况3：无法继续切分特征时，返回叶结点
			cur_node = self.Node(label=self._majority_class(ytrain),
						examples=examples,
						depth=depth)
			if self._cur_model._root is None:
				self._cur_model.add_root(cur_node)
			return	cur_node

		bestFeat,loss = res	
		bestFeatVals = set([example[bestFeat] for example in xtrain])
		
		resNode = self.Node(feature=global_labels[bestFeat],examples=examples,depth=depth)
		del global_labels[bestFeat]
		if self._cur_model._root is None:
			self._cur_model.add_root(resNode)
		else:
			self._cur_model._size += 1
		
		#仅当当前结点深度depth小于限定深度max_depth时才分裂当前结点
		if max_depth is None or depth < max_depth:
			#若分裂结点，则尝试对最优特征的每个值构建子树
			for val in bestFeatVals:
				cur_examples = self._splitDataSet(xtrain,ytrain,bestFeat,val)
				newChild = self._fit(xtrain=cur_examples[0],
							ytrain=cur_examples[1],
							examples=cur_examples,
							global_labels=global_labels,
							depth=resNode._depth+1,
							max_depth=max_depth)
				newChild._parent_split_feature_val = (resNode._feature,val)
				resNode._children[val] = newChild
				newChild._parent = resNode
				self._cur_model._size += 1
		#若当前结点未分裂(深度到达限制),需要设定当前结点为叶结点
		if self._cur_model.num_children(resNode) == 0:
			resNode._feature = None
			resNode._label = self._majority_class(ytrain)	
		resNode._loss = loss
		return resNode	

	def _fixdata(self):
		self._reader._xtrain = np.asarray(self._reader._xtrain,dtype='int64')
		self._reader._xtest = np.asarray(self._reader._xtest,dtype='int64')
		self._reader._xeval = np.asarray(self._reader._xeval,dtype='int64')

	def _get_examples(self,node):
		'''获取node存放的样本'''
		self._cur_model._validate(node)
		return node._examples
	
	def _loss(self,node):
		'''后剪枝时计算叶结点的损失(不包含惩罚项)'''
		_,ydata = self._get_examples(node)
		return self._calInformationEntropy(ydata)

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
		for nd in self._cur_model.preOrder():
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
				0.更新决策树_size属性为原来_size减去新叶结点原来的孩子数量
				1.把父结点入队；
				2.父结点_label设置为所拥有样本的众数类；
				3.清空父结点所有孩子
				4.维护len_before变量为len_after，模型虽已经上提了叶结点，但是原来叶结点nd的
					兄弟结点还在队列中，遍历到这些兄弟结点时，由while循环开头利用new_leafs
					过滤掉即可
				'''
				self._cur_model._size = self._cur_model._size - len(nd._parent._children)
				leafs.append(nd._parent)
				_,ydata = self._get_examples(nd._parent)
				nd._parent._label = self._majority_class(ydata) 
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

		if self._cur_model._root in leafs:
			self._cur_model._size = 1
	
	#---------------------------------公开方法--------------------------------
	def print_tree(self):
		'''层序遍历输出决策树结点及结点关键信息'''
		Q = collections.deque()
		Q.append(self._cur_model._root)
		while len(Q) != 0:
			node = Q.popleft()
			node.showAttributes()
			print('---\n')
			for child in node._children.values():
				Q.append(child)

	def fit(self,xtrain=None,
			ytrain=None,
			examples=None,
			depth=None,
			max_depth=None,
			alpha_leaf=0,
			bool_prune=False):
		"""模型拟合的公开接口。若训练数据集未直接提供，则使用self._reader读取训练数据集
		Args:
			alpha_leaf:后剪枝对叶结点的正则化超参数,有效取值大于等于0.
		"""
		if xtrain is None or ytrain is None:
			self._fixdata()
			self._cur_model = self.DecisionTree()
			self._fit(xtrain=self._reader._xtrain,
					ytrain=self._reader._ytrain,
					examples=(self._reader._xtrain,self._reader._ytrain),
					global_labels=list(range(len(self._reader._xtrain[0]))),
					depth=1,
					max_depth=max_depth)

		else:
			xtrain = self._assert_xdata(xtrain)
			ytrain = self._assert_ydata(ytrain)
			self._cur_model = self.DecisionTree()
			self._fit(xtrain=self._reader._xtrain,
					ytrain=self._reader._ytrain,
					examples=(self._reader._xtrain,self._reader._ytrain),
					global_labels=list(range(len(self._reader._xtrain[0]))),
					depth=1,
					max_depth=max_depth)
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
			cur_xtest = self._assert_xdata(xtest)

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
				raise self.PredictionError('待预测样本 {} 某个属性出现了新的取值!'.format\
					(repr(cur_xtest[i])))
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

	class Node(DecisionTreeClassifierBase.Node):
		'''决策树的结点类'''
		__slots__ = '_feature','_children','_label','_examples','_parent',\
				'_depth','_parent_split_feature_val','_loss'
		def __init__(self,feature=None,label=None,examples=None,parent=None,
				depth=None,parent_split_feat_val=None):
			'''
			Args:
				feature:存储当前结点的划分属性
				label:若为叶结点，则_label属性存储了该叶结点的标签,否则为None
				examples:tuple.每个结点存储了自己拥有的xtrain与ytrain
				parent父结点，根结点设置为None
				depth:结点的深度,根结点设置深度为1
				parent_split_feature_val:父结点划分属性对应本结点的取值
				
			'''
			self._feature = feature
			self._children = collections.OrderedDict()
			#若为叶结点，则_label属性存储了该叶结点的标签
			self._label = label
			self._examples = examples
			self._parent = parent	
			self._depth = depth
			self._parent_split_feature_val = None
			#存储当前最优分裂结点对应的损失值(典型损失函数为信息增益、信息增益比、基尼指数)
			self._loss = None
	
		def showAttributes(self):
			print('_depth:'+repr(self._depth))
			print('父结点划分特征取值_parent_split_feature_val:'+repr(self._parent_split_feature_val))
			print('当前划分属性_feature:'+repr(self._feature))
			print('当前划分属性_loss:'+repr(self._loss))
			#print('_children:'+repr(self._children))
			print('_label:'+repr(self._label))
			#print('_examples:'+repr(self._examples))
			#print('_parent:'+repr(self._parent))

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
			return node._children == {} and node._label != None

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
				if not self.is_leaf(node):
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
			if self.is_leaf(node) or node._children is None:
				return 0
			else:
				return len(node._children)
			
 
		#-----------------------------更新方法------------------------------				
		def add_root(self,node):
			'''为决策树添加根结点，根结点深度设定为1'''
			self._root = node
			node._depth = 1
			self._size = 1


if __name__ == '__main__':
	obj = ID3Classifier(dataDir='/home/michael/data/GIT/MachineLearning/data/forID3')
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
	obj.fit(alpha_leaf=0,max_depth=3,bool_prune=True)
	#obj.fit(alpha_leaf=0,bool_prune=False)
	obj.print_tree()
	print(obj.eval(bool_use_stored_model=False)[0])
	print(obj.eval(bool_use_stored_model=False)[1])
	print(obj._cur_model._size)
