/******************************************************************************
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

#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#define NUM_BINS 16
#define K_BIN (256 / NUM_BINS)

#define ACTIVE_CHANNELS 4
#define NUM_PARTS 64
#define PixelType uchar4

__device__ __forceinline__ void DecodePixel(uchar4 pixel, unsigned int (&bins)[ACTIVE_CHANNELS]);

__global__ void histogram_smem_atomics(
	const PixelType *in,
	int width,
	int height,
	unsigned int *out);

__global__ void histogram_smem_accum(
	const unsigned int *in,
	int n,
	unsigned int *out);

void run_smem_atomics(
	PixelType *d_image,
	int width,
	int height,
	unsigned int *d_hist);

__global__ void histogram_gmem_atomics(
	const PixelType *in,
	int width,
	int height,
	unsigned int *out);

__global__ void histogram_gmem_accum(
	const unsigned int *in,
	int n,
	unsigned int *out);

void run_gmem_atomics(
	PixelType *d_image,
	int width,
	int height,
	unsigned int *d_hist);

#endif
