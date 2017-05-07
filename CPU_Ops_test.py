from __future__ import unicode_literals, print_function

from thinc.neural import ops
import numpy as np
import time
import random

nOps = ops.NumpyOps()

random.seed(5)
words = []
lens = []
alen = 380

for w in range(256):
    alen = alen + 1 
    lens.append(alen)
    for aWord in range(alen):
        vec = []
        for k in range(32):
            vec.append(random.randint(0,200))
        words.append(vec)
    
print(len(lens))

anInput = np.array(words, np.float32)

lengths = np.array(lens, np.int32)

start = time.time()
means = nOps.mean_pool(anInput,lengths)
end = time.time()

print('time elapsed for mean: ', end - start)
print('-------------------------------------')

outfile = 'outmeans.txt'
with open(outfile, 'w') as file:
    file.writelines(' '.join(str(j) for j in i) + '\n' for i in means)

    
means = np.array(means, np.float32)
        
start = time.time()
backprop_words = nOps.backprop_mean_pool(means,lengths)
end = time.time()

print('time elapsed for backprop_mean: ', end - start)
print('-------------------------------------')

outfile = 'outbackpropmeans.txt'
with open(outfile, 'w') as file:
    file.writelines(' '.join(str(j) for j in i) + '\n' for i in backprop_words)



start = time.time()
best,which = nOps.max_pool(anInput,lengths)
end = time.time()

print('time elapsed for max_pool: ', end - start)
print('-------------------------------------')

start = time.time()
backprop_maxes = nOps.backprop_max_pool(best, which,lengths)
end = time.time()

print('time elapsed for backprop_max_pool: ', end - start)
print('-------------------------------------')

outfile = 'outbackpropmeans.txt'
with open(outfile, 'w') as file:
    file.writelines(' '.join(str(j) for j in i) + '\n' for i in backprop_maxes)


print('finished')