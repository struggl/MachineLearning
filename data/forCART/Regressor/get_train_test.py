if __name__ == '__main__':
	path = 'forestfires.csv'
	with open(path,'r',encoding='utf-8') as fr:
		n = 0
		with open('train.tsv','w',encoding='utf-8') as fw:
			for line in fr:
				if n == 0:
					n += 1
					continue
				if n > 300:
					break
				newLine = line.split(',')
				newLine = newLine[:2] + newLine[4:]
				fw.write('\t'.join(newLine))
				n += 1
	with open(path,'r',encoding='utf-8') as fr:
		n = 0
		with open('test.tsv','w',encoding='utf-8') as fw:
			for line in fr:
				if n <= 300:
					n += 1
					continue
				newLine = line.split(',')
				newLine = newLine[:2] + newLine[4:]
				fw.write('\t'.join(newLine))
				n += 1
			
				
