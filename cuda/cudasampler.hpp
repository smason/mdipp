#ifndef cudasampler_hpp
#define cudasampler_hpp

#include <vector>
#include <stdexcept>
#include <iostream>

#ifdef __CUDACC__
std::ostream &operator<<(std::ostream &os, const float2 &p);
#else
// cheekily define these here so I don't need to pull in the full CUDA
// header, hence code "unrelated" to CUDA doesn't need to have the
// CUDA devkit installed
typedef enum cudaError {
  cudaSuccess = 0 // from $CUDA/include/driver_types.h
} cudaError_t;
struct float2 { float x, y; };
#endif

namespace cuda {
  class cudaerror : public std::exception {
    cudaError_t err;

  public:
    cudaerror(const cuda::cudaerror &err) : err(err.err) { showStack(); }
    cudaerror(cudaError_t err) : err(err) { showStack(); }

    void showStack();

    // old @nothrow spec, for compatibility
    const char *what() const throw();

    operator bool() const {
      return err == cudaSuccess;
    }
  };

  // wrap up the CUDA C libray in minimal C++ functions, mainly
  // allowing you to call this without directly including <cuda.h>.
  // these will throw an exception of type @cudaerror on error
  void syncDevice(void);
  void checkErrorsMaybeSyncDevice(void);
  void*cudaMalloc(size_t nbytes);
  void cudaFree(void *ptr);
  void cudaMemcpyDeviceToHost (void *dst, const void *src, size_t nel);
  void cudaMemcpyHostToDevice (void *dst, const void *src, size_t nel);

  // this doesn't throw and just prints to stdout in debug builds,
  // hence designed for destructors
  void cudaFreeNoThrow(void *ptr) throw();

  template<typename T>
  class vector {
    size_t _nel;
    T *_ptr;

    void init(size_t n) {
      if (n > 0) {
	try {
	  _ptr = (T*)cudaMalloc(sizeof(T) * n);
	  _nel = n;
	} catch (...) {
	  _ptr = 0;
	  _nel = 0;
	  throw;
	}
      } else {
	_nel = 0;
	_ptr = 0;
      }
    }

  public:
    vector() : _nel(0), _ptr(0) {
      /*
	should somehow get to a static object that is responsible for 
       */}

    size_t size() const { return _nel; }
    operator T*() const { return _ptr; }

    T get(size_t i) const {
      if (i >= _nel)
	throw std::out_of_range("invalid index");

      T out;
      cudaMemcpyDeviceToHost(&out, _ptr + i, sizeof(T));
      return out;
    }

    void put(size_t i, const T &v) {
      if (i >= _nel)
	throw std::out_of_range("invalid index");

      cudaMemcpyHostToDevice(_ptr + i, &v, sizeof(T));
    }

    explicit vector(size_t n) { init(n); }

    virtual ~vector() {
      if (_nel > 0) {
	// what should I do with errors from this free()?
	cudaFreeNoThrow(_ptr);
	// zero these out for safety
	_ptr = 0;
	_nel = 0;
      }
    };

    void resize(size_t n) {
      if (_ptr) {
	try {
	  cudaFree(_ptr);
	} catch(...) {
 	  // something went wrong, just reset us for now. at some
	  // point, it may be worth checking to see whether this is an
	  // "out of memory" related error before @{throw}ing and
	  // hence failing to reallocate
	  _ptr = 0;
	  _nel = 0;
	  throw;
	}
      }
      init(n);
    }

    operator std::vector<T>() const {
      std::vector<T> ret(_nel);
      cudaMemcpyDeviceToHost(ret.data(), _ptr, sizeof(T) * _nel);
      return ret;
    }

    vector<T>& operator=(const std::vector<T> &src) {
      if (src.size() != _nel)
	throw std::out_of_range("host vector different from device size");
      cudaMemcpyHostToDevice(_ptr, src.data(), sizeof(T) * src.size());
      return *this;
    }
  };

  class sampler {
    friend class dataset;

    const int _nitems, _nclus, _nfiles;
    int _ittr, _nweightcoef, _nsamplers;

    // prefix "_*" for cpu-side variables, "d_*" for device-side

    vector<int>
      d_alloc,  // [nfiles * nitems]
      d_ialloc; // [(nclus+nitems) * nfiles]
    vector<float>
      d_phi, d_lrho,      // [nfiles*(nfiles-1)/2]
      d_weight, d_lweight; // [nfiles * nclus]

    vector<float>
      d_prod,
      d_prodsums,
      d_nu,
      d_dpm;

    vector<float> d_prob; // [nfiles,nclus,nitems]

    vector<int>
      d_phiord,
      d_weightord;

    template<typename OP> void runNuConditional     (float *out, int ittr) const;
    template<typename OP> void runPhiConditionals   (float *out, int ittr) const;
    template<typename OP> void runGammaConditionals (float *out, int ittr) const;

    void allocChanged(const std::vector<int> &alloc);

  public:
    sampler (int nfiles, int nitems, int nclus,
       const std::vector<int> &phiord,
       const std::vector<int> &weightord);
    virtual ~sampler();

    int nitems() const { return _nitems; }
    int nclus()  const { return _nclus;  }
    int nfiles() const { return _nfiles; }

    void setIttr (int gpuittr) { _ittr = gpuittr; }
    void setNu(float nu) { d_nu.put(0, nu); }
    void setDpMass(const std::vector<float> &dpmass) { d_dpm = dpmass; }

    void setAlloc(const std::vector<int> &alloc);
    void setPhis(const std::vector<float> &phis);
    void setWeights(const std::vector<float> &weight);

    std::vector<int>   getAlloc()   const { return d_alloc;  }
    std::vector<float> getWeights() const { return d_weight; }
    std::vector<float> getDpMass()  const { return d_dpm;    }
    std::vector<float> getPhis()    const { return d_phi;    }

    class dataset* nextFileSampler();

    void sampleAlloc();

    void sampleNuConditional ();
    void samplePhiConditionals ();
    void sampleGammaConditionals ();
    void sampleDpMass ();

    float collectNuConditionalAlpha () const;
    std::vector<float> collectPhiConditionalsAlpha () const;
    std::vector<float> collectGammaConditionalsAlpha () const;

    float collectNuConditionalBeta () const;
    std::vector<float> collectPhiConditionalsBeta () const;
    std::vector<float> collectGammaConditionalsBeta () const;
  };

  /** \class{datatype} Define the abstract base class for a \textssc{cuda}
   * sampler of a given datatype.  The only thing we care about is performing a
   * single \ac{MC} step of the cluster parameters, then calculating the
   * (unnormalised) log-probabilities associated with assigning each
   * item to each cluster, and storing these in the device array
   * @d_lprob.
   */
  class dataset {
    cuda::sampler *_gpu;

    int _myfile;

  public:
    dataset (cuda::sampler *gpu, int myfile) : _gpu(gpu), _myfile(myfile) { }

    int myfile() const { return _myfile; }

    int nitems() const { return _gpu->nitems(); }
    int nclus()  const { return _gpu->nclus();  }
    int nfiles() const { return _gpu->nfiles(); }

    int ittr() { return _gpu->_ittr++; }

    // these are pointers into CUDA device memory space
    int*   mygpuinverseallocs() const { return _gpu->d_ialloc + _myfile * (nclus()+nitems()+1); }
    float* mygpuprobs() const { return _gpu->d_prob + _myfile * nclus() * nitems(); }
  };

  struct gaussian {
    dataset *gpu;

    int nfeatures;
    vector<float> d_data;
    vector<float2> d_par;

    void sampleParameters();
    void accumAllocProbs();
  };

  struct gaussianprocess {
    dataset *gpu;

    int nfeatures;
    vector<float> d_data, d_sampfn, d_sigma;

    void accumAllocProbs();
  };

  struct multinomial {
    dataset *gpu;

    int nfeatures, nlevels;
    vector<int>	d_data;
    vector<float> d_lpar;

    void sampleParameters();
    void accumAllocProbs();
  };

  struct bagofwords {
    dataset *gpu;

    int nfeatures;
    vector<int>	d_data;
    vector<float> d_lpar;

    void sampleParameters();
    void accumAllocProbs();
  };
}

#endif // #ifndef cudasampler_hpp
