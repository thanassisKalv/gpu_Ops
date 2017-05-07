
/*
	// Cython function from 'thinc' library
class NumpyOps(Ops):

    def backprop_mean_pool(self, float[:, ::1] d_means, int[::1] lengths):
        cdef int B = lengths.shape[0]
        cdef int O = d_means.shape[1]
        cdef int T = 0
        for length in lengths[:B]:
            T += length
        cdef Pool mem = Pool()
        dX = <float*>mem.alloc(T * O, sizeof(float))

        cpu_backprop_mean_pool(dX,
            &d_means[0,0], &lengths[0], B, T, O)

        return cpu_floats_ptr2array(dX, (T, O))

cdef void cpu_backprop_mean_pool(float* dX__to,
        const float* d_means__bo, const int* lengths__b,
        int B, int T, int O) nogil:
    cdef float scale = 0.
    for length in lengths__b[:B]:
        scale = 1./ length
        for _ in range(length):
            VecVec.add_i(dX__to,
                d_means__bo, scale, O)
            dX__to += O
        d_means__bo += O

*/


void __global__ backprop_mean_pool(float* means, float *words, int *lengths,int *prevLengths, int numdocs, int dims) 
{

    	int bid = blockIdx.x;

	__shared__ float local_means[256];


    	for(int step = bid; step < numdocs; step += gridDim.x )
	{
		int wordsInDoc = lengths[step];
		int blockStarts = prevLengths[step]*dims; 

		local_means[threadIdx.x] = means[step*dims+threadIdx.x];

		for (int i = blockStarts + threadIdx.x; i < blockStarts+(wordsInDoc*dims) ; i += dims)
			words[i] = local_means[threadIdx.x]/wordsInDoc;
		
	}
}

