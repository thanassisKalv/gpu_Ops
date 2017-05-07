

/*
	// Cython function from 'thinc' library
class NumpyOps(Ops):

    def mean_pool(self, float[:, ::1] X, int[::1] lengths):
        cdef int B = lengths.shape[0]
        cdef int O = X.shape[1]
        cdef int T = X.shape[0]

        cdef Pool mem = Pool()
        means = <float*>mem.alloc(B * O, sizeof(float))

        cpu_mean_pool(means,
            &X[0, 0], &lengths[0], B, T, O)
        return cpu_floats_ptr2array(means, (B, O))


cdef void cpu_mean_pool(float* means__bo,
        const float* X__to, const int* lengths__b,
        int B, int T, int O) nogil:
    '''Compute means of a batch of concatenated sequences, using the lengths.'''
    cdef float scale = 0.
    for length in lengths__b[:B]:
        scale = 1. / length
        for _ in range(length):
            VecVec.add_i(means__bo,
                X__to, scale, O)
            X__to += O
        means__bo += O
*/

// hardcoded the shared memory to 256  but we can easily change the host to invoke
// the kernel to dynamically allocate the shared memory (according to vector dimensions) 
void __global__ mean_pool(float* means, float *words, int *lengths,int *prevLengths, int numdocs, int dims) 
{

    	int bid = blockIdx.x;

	__shared__ float local_means[256];

    	for(int step = bid; step < numdocs; step += gridDim.x )
	{
		int wordsInDoc = lengths[step];
		int blockStarts = prevLengths[step]*dims; 

		local_means[threadIdx.x] = 0.0;

		for (int i = blockStarts + threadIdx.x; i < blockStarts+(wordsInDoc*dims) ; i += dims)
			local_means[threadIdx.x] +=  words[i];
		
		__syncthreads();

		means[step*dims + threadIdx.x] = local_means[threadIdx.x]/(float)wordsInDoc;
	}
}


