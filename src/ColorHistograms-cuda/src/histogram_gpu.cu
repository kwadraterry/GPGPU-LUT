#include "histogram.h"
#include <helper_cuda.h>       // CUDA device initialization helper functions

// Decode uchar4 pixel into bins
__device__ __forceinline__ void DecodePixel(uchar4 pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
{
	unsigned char* samples = reinterpret_cast<unsigned char*>(&pixel);

	#pragma unroll
	for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
		bins[CHANNEL] = (unsigned int) (samples[CHANNEL]);
}

// First-pass histogram kernel (binning into privatized counters)
__global__ void histogram_smem_atomics(
	const PixelType *in,
	int width,
	int height,
	unsigned int *out)
{
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

	// initialize smem
	__shared__ unsigned int smem[ACTIVE_CHANNELS * NUM_BINS + 3];
	for (int i = t; i < ACTIVE_CHANNELS * NUM_BINS + 3; i += nt)
		smem[i] = 0;
	__syncthreads();

	// process pixels
	// updates our group's partial histogram in smem
	for (int col = x; col < width; col += nx)
	{
		for (int row = y; row < height; row += ny)
		{
			PixelType pixel = in[row * width + col];

			unsigned int bins[ACTIVE_CHANNELS];
			DecodePixel(pixel, bins);

			#pragma unroll
			for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
				atomicAdd(&smem[(NUM_BINS * CHANNEL) + bins[CHANNEL] + CHANNEL], 1);
		}
	}

	__syncthreads();

	// move to our workgroup's slice of output
	out += g * NUM_PARTS;

	// store local output to global
	for (int i = t; i < NUM_BINS; i += nt)
	{
		#pragma unroll
		for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL) {
			out[i + NUM_BINS * CHANNEL] = smem[i + NUM_BINS * CHANNEL + CHANNEL];
		}
	}
}

// Second pass histogram kernel (accumulation)
__global__ void histogram_smem_accum(
	const unsigned int *in,
	int n,
	unsigned int *out)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > ACTIVE_CHANNELS * NUM_BINS) return; // out of range
	unsigned int total = 0;
	for (int j = 0; j < n; j++)
		total += in[i + NUM_PARTS * j];
	out[i] = total;
}



void run_smem_atomics(
	PixelType *d_image,
	int width,
	int height,
	unsigned int *d_hist)
{

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);

	dim3 block(32, 4);
	dim3 grid(16, 16);
	int total_blocks = grid.x * grid.y;

	// allocate partial histogram
	unsigned int *d_part_hist;
	checkCudaErrors(cudaMalloc(&d_part_hist, total_blocks * NUM_PARTS * sizeof(unsigned int)));

	dim3 block2(128);
	dim3 grid2((ACTIVE_CHANNELS * NUM_BINS + block.x - 1) / block.x);

	histogram_smem_atomics<<<grid, block>>>(
		d_image,
		width,
		height,
		d_part_hist);

	histogram_smem_accum<<<grid2, block2>>>(
		d_part_hist,
		total_blocks,
		d_hist);


	checkCudaErrors(cudaFree(d_part_hist));
}



__global__ void histogram_gmem_atomics(
        const PixelType *in,
        int width,
        int height,
        unsigned int *out)
{
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

	// initialize smem
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


