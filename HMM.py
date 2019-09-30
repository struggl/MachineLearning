'''隐马尔科夫模型(Hidden Markov Model,HMM)'''
import numpy as np

def create_data():
	'''李航<<统计学习方法>>P186维特比解码的例题'''
	#概率转移矩阵A
	A = [[0.5,0.2,0.3],
		[0.3,0.5,0.2],
		[0.2,0.3,0.5]]
	A = np.asarray(A)
	
	#观测概率矩阵，行序号代表状态，列序号代表观测
	B = [[0.5,0.5],
		[0.4,0.6],
		[0.7,0.3]]
	B = np.asarray(B)

	#初始状态概率向量PI
	PI = [0.2,0.4,0.4]
	PI = np.asarray(PI)

	#待解码向量
	seqs = [[0,1,0]]
	seqs = np.asarray(seqs)
	
	return A,B,PI,seqs

###实现Viterbi解码
def Viterbi(A,B,PI,seqs):
	'''维特比解码
	Args:
		A:概率转移矩阵
		B:观测概率矩阵
		PI:初始状态概率向量
		seqs:待解码的观测向量
	'''
	#待预测序列的数量
	nBatch = len(seqs)
	#时间步长度
	T = len(seqs[0])
	#状态集合大小
	N = len(A[0])
	#观测集合大小
	M = len(B[0])
	#预测结果及对应的概率值
	preds = []
	P = []
	for batch in range(nBatch):
		#当前待预测向量及其预测结果
		seq = seqs[batch]
		pred = [None] * T
		#最优路径概率delta，delta为T*N矩阵,行索引t表示时间步，列索引i表示观测为i
		#delta[t][i]表示t时刻观测为i的所有可能路径的概率的最大值
		delta = [None] * T
		#delta_t表示初始时刻t=0，各个状态下产生初始观测的最大概率
		delta[0] = list(PI * B[:,seq[0]])

		#bestNode为T*N维向量,bestNode[t][i]表示时刻t状态为i的所有单个路径中概率最大路径的第t-1个结点
		bestNode = [None] * T
		#初始化bestNode在时刻t=0的取值为None
		bestNode[0] = [None] * N

		for t in range(1,T):
			delta[t] = [None] * T
			bestNode[t] = [None] * T
			for i in range(N):
				#各路径t时刻状态为i的概率,其中最大的那个即为时刻t状态i的概率
				probStateToState = delta[t-1] * A[:,i]
				bestNode[t][i] = np.argmax( probStateToState )
				delta[t][i] = max( probStateToState * B[i,seq[t]] )
		print('---')
		print('batch:{}'.format(batch))
		print('delta:{}'.format(delta))
		print('bestNode:{}'.format(bestNode))
		print('---')
		pred[T-1] = np.argmax(delta[T-1])
		t = T-2
		while t >= 0:
			pred[t] = bestNode[t+1][pred[t+1]]
			t -= 1
		preds.append(pred)	
		P.append(max(delta[T-1]))
	P = np.asarray(P)
	preds = np.asarray(preds)
	return preds,P

if __name__ == '__main__':
	A,B,PI,seqs = create_data()
	preds,P = Viterbi(A,B,PI,seqs)
	print(preds)
	print(P)
