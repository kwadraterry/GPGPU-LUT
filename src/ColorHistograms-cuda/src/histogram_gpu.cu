/******************************************************************************
 * Copyright (c) 2017, Maria Glukhova.  All rights reserved.
 * Copyright (c) 2011-2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include "histogram_gpu.h"
#include "histogram_common.h"
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


__global__ void histogram_strips_accum(
    const unsigned int *in,
    int n_hists,
    int hist_size,
    unsigned int *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > ACTIVE_CHANNELS * NUM_BINS)
        return; // out of range

    unsigned int total = 0;
    for (int hist_id = 0; hist_id < n_hists; hist_id++)
        total += in[hist_size * hist_id + i];

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
	int device_count = 0;
	cudaError_t error_id = cudaGetDeviceCount(&device_count);

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



void run_multigpu(
    PixelType *h_image,
    int width,
    int height,
    unsigned int *h_hist)
{
	printf("run_multigpu\n");
	int device_count = 0;
	cudaError_t error_id = cudaGetDeviceCount(&device_count);
	printf("Found %d cuda-compatible devices.\n", device_count);
    PixelType* d_pixels[device_count];
    unsigned int* d_hist[device_count];
	unsigned int* d_part_hist[device_count];
	unsigned int* d_all_hists;
	unsigned int* d_accumulated_hists;

	// TODO: Check how that work with height % device count != 0 and
	// fix if something is wrong.
    size_t height_d = ceil((double)height / (double)device_count);
    int number_of_bytes_d = sizeof(PixelType) * width * height_d;
    printf("height_d = %lu, nmb = %i\n", height_d, number_of_bytes_d);

	for (int dev_id = 0; dev_id < device_count; dev_id++) {
	    cudaSetDevice(dev_id);
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, 0);
		printf("Device %d allocate 1\n", dev_id);

		// Allocate place in device memory where we will store
		// the 'strip' of the image associated with the device.
		checkCudaErrors(cudaMalloc(
				(void **)&d_pixels[dev_id],
				number_of_bytes_d
		));

		printf("Device %d copy 1\n", dev_id);
		// Copy relevant part of the pixels.
		checkCudaErrors(cudaMemcpy(
				d_pixels[dev_id],
				h_image + number_of_bytes_d * dev_id,
				number_of_bytes_d,
				cudaMemcpyHostToDevice
		));

		dim3 block(32, 4);
		dim3 grid(16, 16);
		int total_blocks = grid.x * grid.y;

	    cudaSetDevice(dev_id);
		printf("Device %d allocate 2\n", dev_id);
		// Allocate partial histogram.
		checkCudaErrors(cudaMalloc(&d_part_hist[dev_id],
				total_blocks * NUM_PARTS * sizeof(unsigned int)));

		printf("Device %d computing partial histograms\n", dev_id);
		// Compute partial histogram.
		histogram_gmem_atomics<<<grid, block>>>(
			d_pixels[dev_id],
			width,
			height_d,
			d_part_hist[dev_id]);

	    cudaSetDevice(dev_id);
		printf("Device %d allocate 3\n", dev_id);
		// Allocate place for the histogram of our 'strip'.
		checkCudaErrors(cudaMalloc(&d_hist[dev_id],
				ACTIVE_CHANNELS * NUM_BINS * sizeof(uint)));

		dim3 block2(128);
		dim3 grid2((3 * NUM_BINS + block.x - 1) / block.x);

		printf("Device %d accumulating partial histograms\n", dev_id);
		// Accumulate partial histograms.
		histogram_gmem_accum<<<grid2, block2>>>(
			d_part_hist[dev_id],
			total_blocks,
			d_hist[dev_id]);

	    cudaSetDevice(dev_id);
		// Deallocate partial histograms.
		cudaFree(d_part_hist[dev_id]);
		// Deallocate pixel values.
		cudaFree(d_pixels[dev_id]);

		// Now we, once again, have partial histograms,
		// but they are stored on different devices.
	}

	// Let's pick one device as master
	int master_id = 0;
	// And task it with gathering and accumulating histograms from all 'strips'.

	cudaSetDevice(master_id);
	// First of all, allocate memory for "gathered" histograms.
	int hist_size = ACTIVE_CHANNELS * NUM_BINS * sizeof(uint);
	printf("Device %d allocate 4 \n", master_id);
	checkCudaErrors(cudaMalloc(&d_all_hists,
			device_count * hist_size));

	// Create stream for transmission
	printf("Device %d creating stream \n", master_id);
	cudaStream_t stream;
	checkCudaErrors(cudaStreamCreate(&stream));

	// Now, every device (including master one) sends its histogram
	// to the master.
	for (int dev_id = 0; dev_id < device_count; dev_id++) {
	    cudaSetDevice(dev_id);
		printf("Device %d enabling peer access\n", dev_id);
	    //checkCudaErrors(cudaDeviceEnablePeerAccess(master_id, 0 ));
		printf("Device %d copying\n", dev_id);
	    checkCudaErrors(cudaMemcpyPeerAsync(
	    		d_all_hists + dev_id * hist_size, // dest addr
	    		master_id, // dest_device
	    		d_hist[dev_id], // src addr
	    		dev_id, // src device
	    		hist_size, // size of data being sent
	    		stream // transmission stream
	    		));
	    // Deallocate 'strip' histogram.
		checkCudaErrors(cudaFree(d_hist[dev_id]));
	}
//
//	cudaSetDevice(master_id);
//
//
	checkCudaErrors(cudaStreamDestroy(stream));
//
//	// Allocate memory for accumulated histogram.
//	checkCudaErrors(cudaMalloc(&d_accumulated_hists, hist_size));
//
//	dim3 block2(128);
//	dim3 grid2((3 * NUM_BINS + 32 - 1) / 32);
//
//	// Accumulate 'strip' histograms.
//	histogram_strips_accum<<<grid2, block2>>>(
//		d_all_hists,
//		device_count,
//		hist_size,
//		d_accumulated_hists);
//
	// Deallocate 'strip' histograms.
	checkCudaErrors(cudaFree(d_all_hists));
//
//	// Copy accumulated histogram back to CPU.
//	checkCudaErrors(cudaMemcpy(h_hist, d_accumulated_hists, hist_size, cudaMemcpyDeviceToHost));
//
//	// Deallocate accumulated histogram.
//	checkCudaErrors(cudaFree(d_accumulated_hists));
}

