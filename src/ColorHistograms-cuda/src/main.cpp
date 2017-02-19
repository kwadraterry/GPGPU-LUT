/**
 * Copyright 2017 Maria Glukhova
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of
 * its contributors may be used to endorse or promote products derived
 * from this software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

// CUDA utilities and system includes
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

// Helper functions
#include <helper_functions.h>  // CUDA SDK Helper functions
#include <helper_cuda.h>       // CUDA device initialization helper functions
#include <helper_cuda_gl.h>    // CUDA device + OpenGL initialization functions

#include <stdlib.h>
#include <stdio.h>
#include <png.h>

#include "imageio.h"
#include "histogram_gpu.h"
#include "histogram_cpu.h"

void probe_image(PixelType* image, png_infop info, unsigned int n) {
	/**
	 * Debug: print every nth pixel.
	 */
	PixelType pixel;
	for (uint y = 0; y < info->height; y += n){
		for (uint x = 0; x < info->width; x += n){
			pixel = image[y * info->width + x];
			unsigned char* samples = reinterpret_cast<unsigned char*>(&pixel);

			printf("[");
			#pragma unroll
			for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL) {
				printf("%4u", (unsigned int) (samples[CHANNEL]));
			}
			printf("]");
		}
		printf("\n");
	}
}


void print_histogram(unsigned int* hist) {
	/**
	 * Print histogram to the stdout.
	 */
    printf("Histogram:\n");
    printf("########### ");
    for (uint bin = 0; bin < NUM_BINS; bin++) {
       printf("[%3u-%3u]", bin * K_BIN, (bin + 1) * K_BIN - 1);
	}
    printf("\n");
    for (uint ch = 0; ch < ACTIVE_CHANNELS; ch++) {
        printf("Channel %u:  ", ch + 1);
        for (uint bin = 0; bin < NUM_BINS; bin++) {
            printf("%8u ", hist[ch * NUM_BINS + bin]);
        }
        printf("\n");
    }
}


unsigned int compare_histograms(const unsigned int* hist_a,
		const unsigned int* hist_b, unsigned int n) {
	/**
	 * Compare two arrays (in our case, histograms).
	 * If they are identical, return 0.
	 * If they are not, return index of the _last_ differing element.
	 */
	while ( --n > 0 && hist_a[n] == hist_b[n]);
	return n;
}


void print_help(char* arg0) {
	/**
	 * Print standard UNIX-style help message.
	 */
	printf("Compute color histogram of PNG image using GPU and CPU.\n\n");
	printf("Usage: %s file.png [options]\n", arg0);
	printf("Options: \n");
	printf("%-30s %-29s", "-t N, --times N", "Run both GPU and CPU "
			"version N times; output averaged time and speed. Default = 1.\n");
	printf("%-30s %-29s", "-h, --help", "Print this message and exit.\n");
	printf("\nReport bugs to siamezzze@gmail.com.\n");
}


int main (int argc, char* argv[]) {
	// Main variables initialization.
	PixelType* h_pixels;
    PixelType* d_pixels;
    unsigned int* d_hist;
    unsigned int* h_hist;
    unsigned int* h_hist2;
    unsigned int* cpu_hist;

	png_infop info;

    StopWatchInterface *h_timer = NULL;
    unsigned int num_runs = 1;

    // Command-line argument parsing.
	if(argc < 2) {
		printf("Error: No input file specified.\n");
		print_help(argv[0]);
		return -1;
	}
	if (strcmp(argv[1], "-h") == 0 or strcmp(argv[1], "--help") == 0) {
		print_help(argv[0]);
		return 0;
	}

	unsigned int arg_it = 2;
	while (arg_it < argc) {
		if (strcmp(argv[arg_it], "-t") == 0 or
				strcmp(argv[arg_it], "--times") == 0) {
			arg_it++;
			num_runs = std::max(1, std::atoi(argv[arg_it]));
		}
		else if (strcmp(argv[arg_it], "-h") == 0 or
				strcmp(argv[arg_it], "--help") == 0) {
			print_help(argv[0]);
			return 0;
		}
		else {
			printf("Unrecognized argument: %s. Use -h or --help for usage "
					"information.\n", argv[arg_it]);
		}
		arg_it++;
	}

    // Read image.
	if(read_png(argv[1], &info, &h_pixels) == PNG_FAILURE) {
		printf("Error reading file (%s).\n", argv[1]);
		return -1;
	}

	size_t number_of_bytes = sizeof(uchar4) * info->width * info->height;

	printf("Image %s loaded (%lu x %lu px.)\n", argv[1], info->height, info->width);

    sdkCreateTimer(&h_timer);

	// CPU
	printf("\nCPU: \n");
    cpu_hist = (uint* )calloc(ACTIVE_CHANNELS * NUM_BINS * sizeof(uint), sizeof(uint));

    sdkResetTimer(&h_timer);

    for (unsigned int i = 0; i < num_runs; i++) {
    	std::fill(cpu_hist, cpu_hist + ACTIVE_CHANNELS * NUM_BINS * sizeof(uint), 0);
        sdkStartTimer(&h_timer);
    	run_cpu(h_pixels, info->width, info->height, cpu_hist);
        sdkStopTimer(&h_timer);
    }
    sdkStopTimer(&h_timer);
    double cpu_avg_secs = 1.0e-3 * (double)sdkGetTimerValue(&h_timer) / (double)num_runs;


    printf("\nrun_cpu() time (average from %u runs) : %.5f sec, %.4f MB/sec\n\n", num_runs, cpu_avg_secs, ((double)number_of_bytes * 1.0e-6) / cpu_avg_secs);



    print_histogram(cpu_hist);


    // Copy data to GPU
	printf("\nGPU: \n");
    printf("Allocating GPU memory and copying input data...");
    checkCudaErrors(cudaMalloc((void **)&d_pixels, number_of_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_hist, ACTIVE_CHANNELS * NUM_BINS * sizeof(uint)));
    checkCudaErrors(cudaMemcpy(d_pixels, h_pixels, number_of_bytes, cudaMemcpyHostToDevice));
    printf("Done\n");

    // Run histogram computing
    printf("Computing histogram...");

    cudaDeviceSynchronize();
    sdkResetTimer(&h_timer);
    sdkStartTimer(&h_timer);

    for (unsigned int i = 0; i < num_runs; i++) {
    	run_gmem_atomics(d_pixels, info->width, info->height, d_hist);
    }

    cudaDeviceSynchronize();
    sdkStopTimer(&h_timer);
    printf("Done\n");
    double d_avg_secs = 1.0e-3 * (double)sdkGetTimerValue(&h_timer) / (double)num_runs;
    // Copy result back to CPU
    printf("Copying result to CPU...");
    h_hist = (uint* )malloc(ACTIVE_CHANNELS * NUM_BINS * sizeof(uint));
    checkCudaErrors(cudaMemcpy(h_hist, d_hist, ACTIVE_CHANNELS * NUM_BINS * sizeof(uint), cudaMemcpyDeviceToHost));
    printf("Done\n");

    printf("\nrun_gmem_atomics() time (average from %u runs) : %.5f sec, %.4f MB/sec\n\n", num_runs, d_avg_secs, ((double)number_of_bytes * 1.0e-6) / d_avg_secs);

    // Print results.
    print_histogram(h_hist);

    unsigned int hists_ne = compare_histograms(cpu_hist, h_hist, ACTIVE_CHANNELS * NUM_BINS);
    if (hists_ne) {
    	printf("Histograms differ!\nChannel %u, bin %u:\nCPU histogram: %3u\nGPU histogram: %3u\n",
    			hists_ne / NUM_BINS, hists_ne % NUM_BINS, cpu_hist[hists_ne], h_hist[hists_ne]);
    }

    // Multiple GPU
    h_hist2 = (uint* )malloc(ACTIVE_CHANNELS * NUM_BINS * sizeof(uint));
	printf("\nMultiple GPUs: \n");
    run_multigpu(h_pixels, info->width, info->height, h_hist2);
    print_histogram(h_hist2);

    printf("Freeing the memory...");
    checkCudaErrors(cudaFree(d_pixels));
    checkCudaErrors(cudaFree(d_hist));
    free(h_pixels);
    free(h_hist);
    free(h_hist2);
    free(cpu_hist);
    printf("Done\n");
	return 0;
}
