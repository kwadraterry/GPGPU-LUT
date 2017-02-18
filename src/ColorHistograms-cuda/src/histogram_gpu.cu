#include "histogram.h"
#include <helper_cuda.h>       // CUDA device initialization helper functions

__device__ __forceinline__ void DecodePixel(uchar4 pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
{
	/**
	 * Decode uchar4 pixel into bins
	 * @param pixel - uchar4 pixel value.
	 * @param bins (output) - Array of ACTIVE_CHANNELS uints representing binned
	 *                        channel value.
	 */
	unsigned char* samples = reinterpret_cast<unsigned char*>(&pixel);

	#pragma unroll
	for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
		bins[CHANNEL] = (unsigned int) (samples[CHANNEL]) / K_BIN;
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
	 * @param out (output)
	 */
	// global position and size
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int nx = blockDim.x * gridDim.x;
	int ny = blockDim.y * gridDim.y;

	// threads in workgroup
	int t = threadIdx.x + threadIdx.y * blockDim.x; // thread index in workgroup, linear in 0..nt-1
	int nt = blockDim.x * blockDim.y; // total threads in workgroup

	// group index in 0..ngroups-1
	int g = blockIdx.x + blockIdx.y * gridDim.x;

	// initialize gmem
	unsigned int *gmem = out + g * NUM_PARTS;
	for (int i = t; i < ACTIVE_CHANNELS * NUM_BINS; i += nt)
		gmem[i] = 0;
	__syncthreads();

	// process pixels (updates our group's partial histogram in gmem)
	for (int col = x; col < width; col += nx)
	{
		for (int row = y; row < height; row += ny)
		{
			PixelType pixel = in[row * width + col];

			unsigned int bins[ACTIVE_CHANNELS];
			DecodePixel(pixel, bins);

			#pragma unroll
			for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
				atomicAdd(&gmem[(NUM_BINS * CHANNEL) + bins[CHANNEL]], 1);
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
    if (i > ACTIVE_CHANNELS * NUM_BINS)
        return; // out of range

    unsigned int total = 0;
    for (int j = 0; j < n; j++)
        total += in[i + NUM_PARTS * j];

    out[i] = total;
}

void run_gmem_atomics(
    PixelType *d_image,
    int width,
    int height,
    unsigned int *d_hist)
{
	/**
	 * Wrapper for GPU histogram computing.
	 * @param in - input image, uchar4 array of continuously placed pixel values.
	 * @param width - int, image width in pixels.
	 * @param height - int, image height in pixels.
	 * @param out (output)
	 */
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    dim3 block(32, 4);
    dim3 grid(16, 16);
    int total_blocks = grid.x * grid.y;

    // allocate partial histogram
    unsigned int *d_part_hist;
    cudaMalloc(&d_part_hist, total_blocks * NUM_PARTS * sizeof(unsigned int));

    dim3 block2(128);
    dim3 grid2((3 * NUM_BINS + block.x - 1) / block.x);

    histogram_gmem_atomics<<<grid, block>>>(
        d_image,
        width,
        height,
        d_part_hist);

    histogram_gmem_accum<<<grid2, block2>>>(
        d_part_hist,
        total_blocks,
        d_hist);

    cudaFree(d_part_hist);
}


