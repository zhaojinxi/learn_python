import cupy as cp

squared_diff = cp.ElementwiseKernel(
    'float32 x, float32 y',
    'float32 z',
    'z = (x - y) * (x - y)',
    'squared_diff')

x = cp.arange(10, dtype=np.float32).reshape(2, 5)
y = cp.arange(5, dtype=np.float32)
squared_diff(x, y)
squared_diff(x, 5)

z = cp.empty((2, 5), dtype=np.float32)
squared_diff(x, y, z)

squared_diff_generic = cp.ElementwiseKernel(
    'T x, T y',
    'T z',
    'z = (x - y) * (x - y)',
    'squared_diff_generic')

squared_diff_generic = cp.ElementwiseKernel(
    'T x, T y',
    'T z',
    '''
        T diff = x - y;
        z = diff * diff;
    ''',
    'squared_diff_generic')

squared_diff_super_generic = cp.ElementwiseKernel(
    'X x, Y y',
    'Z z',
    'z = (x - y) * (x - y)',
    'squared_diff_super_generic')

add_reverse = cp.ElementwiseKernel(
    'T x, raw T y', 'T z',
    'z = x + y[_ind.size() - i - 1]',
    'add_reverse')

l2norm_kernel = cp.ReductionKernel(
    'T x',  # input params
    'T y',  # output params
    'x * x',  # map
    'a + b',  # reduce
    'y = sqrt(a)',  # post-reduction map
    '0',  # identity value
    'l2norm'  # kernel name
)
x = cp.arange(10, dtype=np.float32).reshape(2, 5)
l2norm_kernel(x, axis=1)

add_kernel = cp.RawKernel(r'''
extern "C" __global__
void my_add(const float* x1, const float* x2, float* y) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    y[tid] = x1[tid] + x2[tid];
}
''', 'my_add')
x1 = cupy.arange(25, dtype=cupy.float32).reshape(5, 5)
x2 = cupy.arange(25, dtype=cupy.float32).reshape(5, 5)
y = cupy.zeros((5, 5), dtype=cupy.float32)
add_kernel((5,), (5,), (x1, x2, y))  # grid, block and arguments
y

@cp.fuse()
def squared_diff(x, y):
    return (x - y) * (x - y)

x_cp = cp.arange(10)
y_cp = cp.arange(10)[::-1]
squared_diff(x_cp, y_cp)
x_np = np.arange(10)
y_np = np.arange(10)[::-1]
squared_diff(x_np, y_np)

@cp.fuse()
def sum_of_products(x, y):
    return cupy.sum(x * y, axis=-1)

@cp.fuse(kernel_name='squared_diff')
def squared_diff(x, y):
    return (x - y) * (x - y)