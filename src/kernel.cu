
#include <stdio.h>


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

