## CUDA kernels for some network ops of the NumpyOps class of 'explosion/thinc' library

This is an CUDA implementation of some functions of the referenced library which target to new deep learning models under development
Operations:
* Elementwise mean
* Elementwise max
* Backward pass of elementwise mean
* Backward pass of elementwise max

The scope here is to provide a Cython->CUDA transformation of the ops and furthermore to experiment with a solution callable form Python/Cython. This is achieved using rmcgibbo's recipe of https://github.com/rmcgibbo/npcuda-example 

the requirements for this one:
- python 2.7
- python setuptools, numpy
- nvcc 
- cython for the cython wrapping method

Here the interface's implementation is very simple (and poor) and serves only to demonstrate the Cython-CUDA cooperation.

In the final implementation we want a class with each of these _ops callabe as a methods through python

Literally everything is done in the constructor of a C++ wrapper class callable through Cython in the wrapper.pyx (itself exposed to simple python)
The constructor of the class is calling all of the above 4 operations in a pipeline fashion and the results are returned through pointers to float and integer arrays, already properly allocated in the python calling code.
 
to build the cython stuff
`$ python setup.py install`

The testing script 'GPU_Ops_test.py' is generating a batch of 'documents' according to the Thinc's representation, fills them with some random numbers and then calls the Cython class with the input and output arrays.

For comparison the 'CPU_Ops_test.py' demonstrates the results of the same pipeline using the same seed for the number generation ('thinc' is need to run this one) 


