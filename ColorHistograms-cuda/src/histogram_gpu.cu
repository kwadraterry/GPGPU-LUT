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

    // Allocate partial histogram.
    unsigned int *d_part_hist;
    cudaMalloc(&d_part_hist, total_blocks * NUM_PARTS * sizeof(unsigned int));

    dim3 block2(128);
    dim3 grid2((ACTIVE_CHANNELS * NUM_BINS + block2.x - 1) / block2.x);

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


__global__ void histogram_gmem_atomics1(
        const PixelType *in,
        int width,
        int height,
        unsigned int *out,
        int dev_id,
        int dev_count)
{
    // Global position and size.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;

    // Threads in the workgroup.
    int t = threadIdx.x + threadIdx.y * blockDim.x; // thread index in workgroup, linear in 0..nt-1
    int nt = blockDim.x * blockDim.y; // total threads in workgroup

    // Group index in 0..ngroups-1.
    int g = blockIdx.x + blockIdx.y * gridDim.x;

    // Initialize global memory.
    unsigned int *gmem = out + g * NUM_PARTS;
    for (int i = t; i < ACTIVE_CHANNELS * NUM_BINS; i += nt)
    gmem[i] = 0;
    __syncthreads();

    // Process pixels (updates our group's partial histogram in gmem).
    for (int col = x; col < width; col += nx)
    {
        for (int row = y; row < height; row += ny)
        {
            PixelType pixel = in[row * width + col];

            unsigned int bins[ACTIVE_CHANNELS];
            DecodePixel(pixel, bins);

            // Every device process its own channel(s).
            #pragma unroll
            for (int CHANNEL = dev_id; CHANNEL < ACTIVE_CHANNELS; CHANNEL += dev_count) {
                atomicAdd(&gmem[(NUM_BINS * CHANNEL) + bins[CHANNEL]], 1);
            }
        }
    }
}


void run_multigpu(
        PixelType *d_image,
        int width,
        int height,
        unsigned int *d_hist,
        int device_id,
        int device_count)
{
    dim3 block(32, 4);
    dim3 grid(16, 16);
    int total_blocks = grid.x * grid.y;

    // Allocate partial histogram.
    // Actually, we need less memory (only the channels assigned to the device),
    // but for the sake of simplicity, let us have the "full" histogram for
    // every device (still counting only relevant bits).
    // TODO: Memory-efficient way.
    unsigned int *d_part_hist;
    cudaMalloc(&d_part_hist, total_blocks * NUM_PARTS * sizeof(unsigned int));

    dim3 block2(128);
    dim3 grid2((ACTIVE_CHANNELS * NUM_BINS + block2.x - 1) / block2.x);

    histogram_gmem_atomics1<<<grid, block>>>(
        d_image,
        width,
        height,
        d_part_hist,
        device_id,
        device_count
        );

    histogram_gmem_accum<<<grid2, block2>>>(
        d_part_hist,
        total_blocks,
        d_hist);


    cudaFree(d_part_hist);
}
