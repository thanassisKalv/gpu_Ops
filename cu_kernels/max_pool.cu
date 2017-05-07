
/*
	// Cython function from 'thinc' library
class NumpyOps(Ops):

    def max_pool(self, float[:, ::1] X, int[::1] lengths):
        cdef int B = lengths.shape[0]
        cdef int O = X.shape[1]
        cdef int T = X.shape[0]

        cdef Pool mem = Pool()
        maxes = <float*>mem.alloc(B * O, sizeof(float))
        which = <int*>mem.alloc(B * O, sizeof(int))

        cpu_max_pool(maxes, which,
            &X[0, 0], &lengths[0], B, T, O)

        cdef ndarray py_best = cpu_floats_ptr2array(maxes, (B, O))
        cdef ndarray py_which = cpu_ints_ptr2array(which, (B, O))
        return py_best, py_which

cdef void cpu_max_pool(float* maxes__bo, int* which__bo,
        const float* X__to, const int* lengths__b,
        int B, int T, int O) nogil:
    '''Compute maxes of a batch of concatenated sequences, using the lengths.'''
    cdef float scale = 0.
    for length in lengths__b[:B]:
        memcpy(maxes__bo, X__to, O * sizeof(maxes__bo[0]))
        memset(which__bo, 0, O * sizeof(which__bo[0]))
        X__to += O
        for i in range(1, length):
            for j in range(O):
                if X__to[j] > maxes__bo[j]:
                    maxes__bo[j] = X__to[j]
                    which__bo[j] = i
            X__to += O
        maxes__bo += O
        which__bo += O

*/


void __global__ max_pool(float* maxes, int* which, float *words, int *lengths,int *prevLengths, int numdocs, int dims) 
{
    	int bid = blockIdx.x;

	__shared__ float local_maxes[256];
	__shared__ short local_which[256];

    	for(int step = bid; step < numdocs; step += gridDim.x )
	{
		int wordsInDoc = lengths[step];
		int blockStarts = prevLengths[step]*dims; 

		local_maxes[threadIdx.x] = words[blockStarts+threadIdx.x];
		local_which[threadIdx.x] = 0;
		short j=1;	// the word index in a doc

		for (int i = blockStarts+dims+threadIdx.x; i < blockStarts+(wordsInDoc*dims) ; i += dims)
		{
			if(words[i]>local_maxes[threadIdx.x])
			{
				local_maxes[threadIdx.x] =  words[i];
				local_which[threadIdx.x] = j;
			}
			j++; 
		}
		__syncthreads();

		maxes[step*dims + threadIdx.x] = local_maxes[threadIdx.x];
		which[step*dims + threadIdx.x] = local_which[threadIdx.x];
	}
}
