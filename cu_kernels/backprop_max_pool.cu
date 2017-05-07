
/*
	// Cython function from 'thinc' library
class NumpyOps(Ops):

    def backprop_max_pool(self, float[:, ::1] d_maxes,
            int[:, ::1] which, int[::1] lengths):
        cdef int B = lengths.shape[0]
        cdef int O = d_maxes.shape[1]
        cdef int T = 0
        for length in lengths[:B]:
            T += length
        cdef Pool mem = Pool()
        dX = <float*>mem.alloc(T * O, sizeof(float))

        cpu_backprop_max_pool(dX,
            &d_maxes[0,0], &which[0, 0], &lengths[0], B, T, O)

        return cpu_floats_ptr2array(dX, (T, O))


cdef void cpu_backprop_max_pool(float* dX__to,
        const float* d_maxes__bo, const int* which__bo, const int* lengths__b,
        int B, int T, int O) nogil:
    cdef int length, i, j
    for length in lengths__b[:B]:
        for i in range(length):
            for j in range(O):
                if which__bo[j] == i:
                    dX__to[j] += d_maxes__bo[j]
            dX__to += O
        d_maxes__bo += O
        which__bo += O
*/


void __global__ backprop_max_pool(float* maxes, int* which, float *words, int *lengths,int *prevLengths, int numdocs, int dims) 
{
    	int bid = blockIdx.x;

	__shared__ float local_maxes[256];
	__shared__ short local_which[256];

    	for(int step = bid; step < numdocs; step += gridDim.x )
	{
		int wordsInDoc = lengths[step];
		int blockStarts = prevLengths[step]*dims; 

		local_maxes[threadIdx.x] = maxes[step*dims+threadIdx.x];
		local_which[threadIdx.x] = which[step*dims+threadIdx.x];
		short j=0;	// the word index in a doc

		for (int i = blockStarts+threadIdx.x; i < blockStarts+(wordsInDoc*dims) ; i += dims)
		{
			if(local_which[threadIdx.x]==j)
			{
				words[i] =  local_maxes[threadIdx.x];
			}
			else
				words[i]=0;
			j++; 
		}

	}
}

