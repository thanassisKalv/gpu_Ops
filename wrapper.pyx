import numpy as np
cimport numpy as np


cdef extern from "src/manager.hh":
	cdef cppclass C_GPU_Ops "GPU_Ops":
		C_GPU_Ops(np.float32_t*, np.float32_t*, np.float32_t*, np.float32_t*, np.int32_t*, np.float32_t*, np.int32_t*, np.int32_t*, int, int)


cdef class GPU_Ops:
	cdef C_GPU_Ops* g
	cdef int dim1
	cdef int dim2
	
	def __cinit__(self, np.ndarray[ndim=1, dtype=np.float32_t] means, 
			np.ndarray[ndim=1, dtype=np.float32_t] words,
			np.ndarray[ndim=1, dtype=np.float32_t] backprop_words,
			np.ndarray[ndim=1, dtype=np.float32_t] maxes,
			np.ndarray[ndim=1, dtype=np.int32_t] which,
			np.ndarray[ndim=1, dtype=np.float32_t] backMax_words, 
			np.ndarray[ndim=1, dtype=np.int32_t] lengths, 
			np.ndarray[ndim=1, dtype=np.int32_t] prevlengths):
		self.dim1 = len(lengths)
		self.dim2 = len(means)/self.dim1
		self.g = new C_GPU_Ops(&means[0], &words[0], &backprop_words[0], &maxes[0], &which[0], &backMax_words[0], &lengths[0], &prevlengths[0], self.dim1, self.dim2)




