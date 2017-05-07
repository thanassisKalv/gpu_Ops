class GPU_Ops {

  float* means_device;
  float *words_device;
  float *backPMean_words_device;
  float *backPMax_words_device;
  int *lengths_device;
  int *prevLengths_device;
  float *maxes_device;
  int *which_device;
  int numdocs;
  int dims;

  // host pointers 
  float* means_host;
  float *words_host;
  float *maxes_host;
  int *which_host;
  float *backPMean_words_host;
  float *backMax_words_host;
  int *lengths_host;
  int *prevLengths_host; 

  // the total size in bytes of the 1D vectors --> words_host[] & words_device[]
  int full_size;

public:


  GPU_Ops(float* MEANS_host,float* WORDS_host, float* BPROP_WORDS_host,  float* MAXES_host, int* WHICH_host, float* BACKMAXWORDS_host_, int *LENGTHS_host, int *PREVLENGTHS_host, int NUMDOCS, int DIMS); // constructor (copies to GPU)

  ~GPU_Ops(); // destructor


  // the constructor



};
