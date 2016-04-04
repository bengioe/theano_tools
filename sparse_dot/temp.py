
class DualVectorNonzero(theano.sandbox.cuda.GpuOp):
    """
    This OP returns two things:
      - the equivalent of T.stack(*A.nonzero()), D
           [[row pos 0, row pos 1, ...]
            [col pos 0, col pos 1, ...]]
      - a coded nonzero index matrix C

    The vector version only returns the indexes of the nonzero indices.
    

    C is defined as such:
    [ [ start position of non-zeros in D for row 0, number of non-zeros for row 0]
      ...
      ...]
    
    """
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return "GPUVectorNonzero"

    def make_node(self, A):
        assert A.ndim == 1
        A = safe_to_gpu(A)
        return theano.Apply(self,
                            inputs=[A],
                            outputs=[A.type()])#T.TensorType('int32', [False, False])()])
    
    def c_support_code(self, *args, **kwargs):
        return """int __warned_transposed = 0;"""

    def c_code(self, node, name, inp, out, sub):
        A, = inp
        B, = out
        fail = sub['fail']
        return """

cudaError_t sts;
float * gpu_A = CudaNdarray_DEV_DATA(%(A)s);
float * gpu_B;
const int * A_dims = CudaNdarray_HOST_DIMS(%(A)s);
int A_size = A_dims[0];
float * cpu_A = (float*)malloc(sizeof(float) * A_size); 
cudaMemcpy(cpu_A, gpu_A, sizeof(float) * A_size, cudaMemcpyDeviceToHost);

int n_nonzeros = 0;
int* nonzero_positions = (int*)malloc(A_size * sizeof(int));
int* nzp_xptr = nonzero_positions;


float *Aptr = cpu_A;
for (int i=0; i<A_size; i++){
  if (*Aptr++ != 0.0){
    *nzp_xptr++ = i;
    n_nonzeros++;
  }
}

free(cpu_A);

int B_dims[] = {n_nonzeros};

if (CudaNdarray_prep_output(&%(B)s, 1, B_dims)){
  %(fail)s;
 }

gpu_B = CudaNdarray_DEV_DATA(%(B)s);
cudaMemcpy(gpu_B, nonzero_positions, sizeof(int) * n_nonzeros, 
           cudaMemcpyHostToDevice);
free(nonzero_positions);


sts = cudaGetLastError();
if (cudaSuccess != sts){
  PyErr_Format(PyExc_RuntimeError,
	       "Cuda error: gpunonzero: %%s",
	       cudaGetErrorString(sts));
            %(fail)s;
 }


        """ % locals()



class _GPUSparseGemv_SparseBySparse(theano.sandbox.cuda.GpuOp):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, *args):
        A,_,B,Omask,__,block_size = args
        aindexes = _
        indexes = __
            
        assert indexes.ndim == 1
        assert aindexes.ndim == 1
        assert A.ndim == 1
        assert B.ndim == 2

        self.block_size = block_size
        if indexes.dtype!='float32':
            indexes = T.cast(indexes, 'float32')
        if A.dtype!='float32':
            A = T.cast(A, 'float32')
        if B.dtype!='float32':
            B = T.cast(B, 'float32')
        A,B,indexes,aindexes = [safe_to_gpu(i) for i in [A,B,indexes,aindexes]]
        return theano.Apply(self,
                            inputs=[A,aindexes,
                                    B,indexes,
                                    Omask],
                            outputs=[theano.sandbox.cuda.CudaNdarrayType(broadcastable=(0,))()])


    def grad(self, inputs, outputs):
        a,ia,b,i,dropout = inputs
        gz, = outputs
        
        gzo = (gz.reshape((dropout.shape[0], -1)) * dropout.dimshuffle(0,'x')).reshape(gz.shape)
        xgrad = T.dot(gzo, b.T)
        ygrad = T.outer(a.T, gzo)

        # MY CONCLUSION is that the gradient wrt to the dropout is
        # defined, but is zero since the dropout matrix should really
        # be of type "boolean", which is like a step function (0 derivative)
        dgrad = T.zeros_like(T.cast(dropout,'float32'))
        disc = theano.gradient.DisconnectedType()
        return tuple([xgrad,disc(),ygrad] + 
                     [disc() for i in [i]]+
                     [disc()])
    def connection_pattern(self, node): 
        # a, aidx, 
        # b, bidx, 
        # dropout
        return [[True], [False], 
                [True], [False],
                [False]]

    def c_support_code(self):
        return """

__global__ void sparsedot_ss(float* A, float* B, int* indexes, int* aindexes, 
        float* C, int n, int m,
        int n_indexes, int n_aindexes,
        int b_stride0, int b_stride1) {
        /*
        A is n
        B is n,m
        indexes is n_indexes
        C is m
        */
#define BLOCK_SIZE %d
#define NTHREADS 1
    __shared__ float Abuf0[BLOCK_SIZE];
    int index_position = blockIdx.x * NTHREADS + threadIdx.x;
    if (index_position >= n_indexes) return;
    int tx = threadIdx.y;

    // this is the position of the computed value in C
    int posx = indexes[index_position] * BLOCK_SIZE + threadIdx.y;

    double acc0 = 0;
    for (int ks=0;ks<n_aindexes;ks++){
        int aposx = aindexes[ks] * BLOCK_SIZE;
        Abuf0[tx] = A[tx + aposx];
        __syncthreads();
        #pragma unroll BLOCK_SIZE
        for (int k=0;k<BLOCK_SIZE;++k){
            double a = Abuf0[k];
            double b = B[((k+aposx) * b_stride0) + (posx * b_stride1)];
            acc0 += a * b;
        }
        __syncthreads();
    }
    C[posx] = acc0;
    
}""" % (self.block_size)


    def c_code(self, node, name, inp, out, sub):
        A,aindexes,B,indexes,dropout = inp
        z, = out
        fail = sub['fail']
        s = """
        float *A_data = CudaNdarray_DEV_DATA(%(A)s);
        float *B_data = CudaNdarray_DEV_DATA(%(B)s);
        int *I_data   = (int*)CudaNdarray_DEV_DATA(%(indexes)s);
        int *AI_data  = (int*)CudaNdarray_DEV_DATA(%(aindexes)s);
        int dims[] = {0,0,0};
        float *O_data; // output data

        const int* A_dims = CudaNdarray_HOST_DIMS(%(A)s);
        const int* B_dims = CudaNdarray_HOST_DIMS(%(B)s);

        int total_size = B_dims[1] * sizeof(float);
        int zdims[] = {B_dims[1]};
        //printf("%%d zdim\\n",B_dims[1]);

        const int* index_dims = CudaNdarray_HOST_DIMS(%(indexes)s);
        const int* aindex_dims = CudaNdarray_HOST_DIMS(%(aindexes)s);
        const int* B_strides = CudaNdarray_HOST_STRIDES(%(B)s);
        

        int grid_size = index_dims[0] / NTHREADS;
        dim3 blocks(NTHREADS, BLOCK_SIZE);

        // we need to round up grid_size, so add one unless excatly a fit (modulo is zero)
        if (index_dims[0] %% (NTHREADS) != 0) grid_size++;

        cudaError_t sts;
        void * orig_z = %(z)s;
        if (CudaNdarray_prep_output(&%(z)s, 1, zdims))
        {
            %(fail)s;
        }
        //printf("0-fill %%d\\n",total_size);
        sts = cudaMemset(CudaNdarray_DEV_DATA(%(z)s), 0, total_size);
        if (cudaSuccess != sts)
        {
            PyErr_Format(PyExc_MemoryError,
                         "GpuEye: Error in memset %%d bytes of device memory.",
                         total_size);
            if(orig_z == NULL)
                Py_XDECREF(%(z)s);
            %(fail)s;
        }
        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
        O_data = CudaNdarray_DEV_DATA(%(z)s);

  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventRecord(start,0);
        if (grid_size > 0){
        sparsedot_ss<<<grid_size, blocks>>>(A_data, B_data, I_data, AI_data, O_data,
                                            A_dims[0], B_dims[1], index_dims[0], aindex_dims[0],
                                            B_strides[0], B_strides[1]); 
        }
        CNDA_THREAD_SYNC;

 cudaEventCreate(&stop);
 cudaEventRecord(stop,0);
 cudaEventSynchronize(stop);

 cudaEventElapsedTime(&elapsedTime, start,stop);
        //printf("Elapsed time : %%f ms %%d %%d %%d %%d (%%d)\\n" ,elapsedTime, grid_size, blocks.x, blocks.y, blocks.z, BLOCK_SIZE);
  /*printf("sparse_dot: %%s. n=%%d, m=%%d. grid=(%%d %%dx%%d), indexes=(%%d %%d), bstrides=(%%d %%d)\\n",
                    cudaGetErrorString(sts),
                    A_dims[0], B_dims[1], grid_size, blocks.x, blocks.y,
                    index_dims[0], index_dims[1],
                    B_strides[0], B_strides[1]);*/
        
        sts = cudaGetLastError();
        if (cudaSuccess != sts)
        {
               PyErr_Format(PyExc_RuntimeError,
                    "Cuda error: sparse_dot: %%s. n=%%d, m=%%d. grid=(%%d %%dx%%d), indexes=(%%d %%d), bstrides=(%%d %%d)\\n",
                    cudaGetErrorString(sts),
                    A_dims[0], B_dims[1], grid_size, blocks.x, blocks.y,
                    index_dims[0], index_dims[1],
                    B_strides[0], B_strides[1]);
            %(fail)s;
        }
        """ % locals()
        return s

    #def c_code_cache_version(self):
    #    return (2,self.block_size)

    def __str__(self):
        return "GPUSparseGemv_SS"










