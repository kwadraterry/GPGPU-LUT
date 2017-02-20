from pycuda.compiler import SourceModule

mod = SourceModule("""
// These are fine to experiment with. They actually would look better as parameters,
//      but I am too lazy for this.
#define NUM_BINS 16         // Number of bins.
#define ACTIVE_CHANNELS 4   // Number of channels for which the histogram is computed.
                            //    Note: PNG has 4 channels (R, G, B, Alpha).
                            //    So, to compute histograms only for RGB, without
                            //    alpha-channel, set this to 3.


// These are for inner use. Nothing stops you from modifying these, too,
//    but that would probably break something.
#define PixelType uchar4                     // Type used for pixel data.
#define K_BIN (256 / NUM_BINS)               // Number of colors stored in one bin.
#define NUM_PARTS NUM_BINS * ACTIVE_CHANNELS // Size of partial histogram.



__device__ __forceinline__ void DecodePixel(
        uchar4 pixel,
        unsigned int (&bins)[ACTIVE_CHANNELS])
{
    /**
    * Decode uchar4 pixel into bins.
    * @param pixel - uchar4 pixel value.
    * @param bins (output) - Array of ACTIVE_CHANNELS uints representing binned
    *                        channel value.
    */
    unsigned char* samples = reinterpret_cast<unsigned char*>(&pixel);

    #pragma unroll
    for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL) {
        bins[CHANNEL] = (unsigned int) (samples[CHANNEL]) / K_BIN;
    }
}

__global__ void histogram_gmem_atomics(
        const PixelType *in,
        int width,
        int height,
        unsigned int *out)
{
    /**
    * First-pass histogram kernel (binning into privatized counters)
    * @param in - input image, uchar4 array of continuously placed pixel values.
    * @param width - int, image width in pixels.
    * @param height - int, image height in pixels.
    * @param out (output) - partial histograms.
    */
    // Global position and size.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;

    // Threads in workgroup.
    int t = threadIdx.x + threadIdx.y * blockDim.x; // thread index in workgroup, linear in 0..nt-1
    int nt = blockDim.x * blockDim.y; // total threads in workgroup

    // Group index in 0..ngroups-1.
    int g = blockIdx.x + blockIdx.y * gridDim.x;

    // Initialize global memory.
    unsigned int *gmem = out + g * NUM_PARTS;
    for (int i = t; i < ACTIVE_CHANNELS * NUM_BINS; i += nt){
        gmem[i] = 0;
    }
    __syncthreads();

    // Process pixels (updates our group's partial histogram in gmem).
    for (int col = x; col < width; col += nx)
    {
        for (int row = y; row < height; row += ny)
        {
            PixelType pixel = in[row * width + col];

            unsigned int bins[ACTIVE_CHANNELS];
            DecodePixel(pixel, bins);

            #pragma unroll
            for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL) {
                atomicAdd(&gmem[(NUM_BINS * CHANNEL) + bins[CHANNEL]], 1);
            }
        }
    }
}

__global__ void histogram_gmem_accum(
        const unsigned int *in,
        int n,
        unsigned int *out)
{
    /**
    * Accumulate partial histograms into global one.
    * @param in - input partial histograms.
    * @param n - total number of blocks.
    * @param out (output) - global histogram.
    */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > ACTIVE_CHANNELS * NUM_BINS) {
        return; // out of range
    }

    unsigned int total = 0;
    for (int j = 0; j < n; j++) {
        total += in[i + NUM_PARTS * j];
    }

    out[i] = total;
}
""")

histogram_atomics = mod.get_function("histogram_gmem_atomics")
histogram_accum = mod.get_function("histogram_gmem_accum")