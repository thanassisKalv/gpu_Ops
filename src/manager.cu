/*
This is the central piece of code, the interface of GPU_ops
This class will get translated into python
*/

#include "kernel.cu"
#include <manager.hh>
#include <assert.h>
#include <iostream>
#include <cuda.h>
using namespace std;


GPU_Ops::GPU_Ops (float* means_host_,float* words_host_, float* bProp_words_, float* maxes_host_, int* which_host_, float* backMax_words_host_,
 				int* lengths_host_, int* prevLengths_host_,int numdocs_,int dims_) 
{

  means_host = means_host_;
  words_host = words_host_;
  lengths_host = lengths_host_;
  prevLengths_host = prevLengths_host_; 
  numdocs = numdocs_;
  dims = dims_;
  backPMean_words_host = bProp_words_;
  maxes_host = maxes_host_;
  which_host = which_host_;
  backMax_words_host = backMax_words_host_;
 
  int full_size = (prevLengths_host[numdocs-1]+lengths_host[numdocs-1]) * dims *sizeof(float);
  cudaError_t err; 

  err = cudaMalloc((void**) &words_device, full_size);
  assert(err == 0);

  err = cudaMalloc((void**) &backPMean_words_device, full_size);
  assert(err == 0);
  
  err = cudaMalloc((void**) &means_device, numdocs*dims*sizeof(float));
  assert(err == 0);

  err = cudaMalloc((void**) &lengths_device, numdocs*sizeof(int));
  assert(err == 0);
  
  err = cudaMalloc((void**) &prevLengths_device, numdocs*sizeof(int));
  assert(err == 0);

  err = cudaMalloc((void**) &which_device, numdocs*dims*sizeof(int));
  assert(err == 0);
  
  err = cudaMalloc((void**) &maxes_device, numdocs*dims*sizeof(float));
  assert(err == 0);

  err = cudaMalloc((void**) &backPMax_words_device, full_size);
  assert(err == 0);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  // copying data from HostToDevice
  err = cudaMemcpy(words_device, words_host, full_size, cudaMemcpyHostToDevice);
  assert(err == 0);

  err = cudaMemcpy(lengths_device, lengths_host, numdocs*sizeof(int), cudaMemcpyHostToDevice);
  assert(err == 0);

  err = cudaMemcpy(prevLengths_device, prevLengths_host, numdocs*sizeof(int), cudaMemcpyHostToDevice);
  assert(err == 0);

  mean_pool<<<32, dims>>>(means_device, words_device, lengths_device, prevLengths_device, numdocs, dims);

  //cudaDeviceSynchronize();

  err = cudaGetLastError();

  if(err != 0) 
  {  cout << "cuda kernel returned error: "<< cudaGetErrorString(err) << endl; assert(0); }


  backprop_mean_pool<<<32, dims>>>(means_device, backPMean_words_device, lengths_device, prevLengths_device, numdocs, dims);

  err = cudaGetLastError();

  if(err != 0) 
  {  cout << "cuda kernel returned error: "<< cudaGetErrorString(err) << endl; assert(0); }

  max_pool<<<32, dims>>>(maxes_device, which_device, words_device, lengths_device, prevLengths_device, numdocs, dims);

  err = cudaGetLastError();

  if(err != 0) 
  {  cout << "cuda kernel returned error: "<< cudaGetErrorString(err) << endl; assert(0); }

  backprop_max_pool<<<32, dims>>>(maxes_device, which_device, backPMax_words_device, lengths_device, prevLengths_device, numdocs, dims);

  err = cudaGetLastError();

  if(err != 0) 
  {  cout << "cuda kernel returned error: "<< cudaGetErrorString(err) << endl; assert(0); }

  cudaMemcpy(means_host, means_device, numdocs*dims*sizeof(float), cudaMemcpyDeviceToHost);
  err = cudaGetLastError();
  if(err != 0) 
  	{  cout << cudaGetErrorString(err) << endl; assert(0); }

  cudaMemcpy(backPMean_words_host, backPMean_words_device, full_size, cudaMemcpyDeviceToHost);
  err = cudaGetLastError();
  if(err != 0) 
  	{  cout << cudaGetErrorString(err) << endl; assert(0); }

  cudaMemcpy(which_host, which_device, numdocs*dims*sizeof(int), cudaMemcpyDeviceToHost);
  err = cudaGetLastError();
  if(err != 0) 
  	{  cout << cudaGetErrorString(err) << endl; assert(0); }

  cudaMemcpy(backMax_words_host, backPMax_words_device, full_size, cudaMemcpyDeviceToHost);
  err = cudaGetLastError();
  if(err != 0) 
  	{  cout << cudaGetErrorString(err) << endl; assert(0); }

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("time elapsed for gpu_mean_pool(): %f milliseconds\n", milliseconds);

}



GPU_Ops::~GPU_Ops() {
  cudaFree(words_device);
  cudaFree(backPMean_words_device);
  cudaFree(lengths_device);
  cudaFree(prevLengths_device);
  cudaFree(backPMax_words_device);
  cudaFree(means_device);
}
