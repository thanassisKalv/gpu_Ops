import gpu_ops
import numpy as np
import numpy.testing as npt
import random
import time

def test():

    	words = []
    	lens = []
	alen = 380
	random.seed(5)
	dim = 32

    	for w in range(256):
        	alen = alen + 1
    		lens.append(alen)
    		for aWord in range(alen):
        		vec = []
        		for k in range(dim):
				vec.append(random.randint(0,200))
			words.append(vec)	
	

	words = np.array(words, dtype=np.float32)
	bMeanProp_words = np.zeros(words.shape[1]*words.shape[0], dtype=np.float32)
	bMaxProp_words = np.zeros(words.shape[1]*words.shape[0], dtype=np.float32)
	lens = np.array(lens, dtype=np.int32)
	means = np.zeros(words.shape[1]*len(lens), dtype=np.float32)
	maxes = np.zeros(words.shape[1]*len(lens), dtype=np.float32)
	which = np.zeros(words.shape[1]*len(lens), dtype=np.int32)
	prevlens = np.zeros(len(lens), dtype=np.int32)

	words = words.ravel()	

	c = 0
	for alen in lens:
		c = c + 1
		prevlens[c] = alen + prevlens[c-1]
		if c == (len(lens)-1):
			break

	#print(prevlens)

	start = time.time()
	gpu_ops.GPU_Ops(means, words, bMeanProp_words, maxes, which, bMaxProp_words, lens, prevlens)
	end = time.time()
    

	outfile = 'gpu_Means.txt'
	afile = open(outfile, 'w')
	for i in range(256):
		for j in range(dim):
			afile.write(str(means[i*dim + j])+' ')
		afile.write('\n')

	#outfile = 'outbackPropmeans.txt'
	#afile = open(outfile, 'w')
	#for i in range(len(words)/dim):
	#	for j in range(dim):
	#		afile.write(str(bProp_words[i*dim + j])+' ')
	#	afile.write('\n')

	outfile = 'gpu_WhichIndex.txt'
	afile = open(outfile, 'w')
	for i in range(256):
		for j in range(dim):
			afile.write(str(which[i*dim + j])+' ')
		afile.write('\n')

	outfile = 'gpu_backPropMaxes.txt'
	afile = open(outfile, 'w')
	for i in range(len(words)/dim):
		for j in range(dim):
			afile.write(str(bMaxProp_words[i*dim + j])+' ')
		afile.write('\n')

	print('time elapsed for gpu_mean_pool(): ', end - start)

test()
