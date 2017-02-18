
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

void probe_image(PixelType* image, png_infop info) {
	/**
	 * Debug: print every 10th pixel.
	 */
	PixelType pixel;
	for (uint y = 0; y < info->height; y+=10){
		for (uint x = 0; x < info->width; x+=10){
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
        printf("Channel %u: ", ch + 1);
        for (uint bin = 0; bin < NUM_BINS; bin++) {
            printf("%8u ", hist[ch * NUM_BINS + bin]);
        }
        printf("\n");
    }
}


int main (int argc, char* argv[]) {
	PixelType* h_pixels;
    PixelType* d_pixels;
    unsigned int* d_hist;
    unsigned int* h_hist;
    unsigned int* cpu_hist;

	png_infop info;

	if(argc < 2) {
		printf("Must include file name to process. `%s <file_name>`\n", argv[0]);
		return -1;
	}

    // Read image.
	if(read_png(argv[1], &info, &h_pixels) == PNG_FAILURE) {
		printf("Error reading file (%s).\n", argv[1]);
		return -1;
	}

	size_t number_of_bytes = sizeof(uchar4) * info->width * info->height;

	printf("Image has height %lu and width %lu\n", info->width, info->width);


    // Copy data to GPU
	printf("GPU: \n");
    printf("Allocating GPU memory and copying input data\n");
    checkCudaErrors(cudaMalloc((void **)&d_pixels, number_of_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_hist, ACTIVE_CHANNELS * NUM_BINS * sizeof(uint)));
    checkCudaErrors(cudaMemcpy(d_pixels, h_pixels, number_of_bytes, cudaMemcpyHostToDevice));

    // Run histogram computing
    printf("Computing histogram\n");
    run_gmem_atomics(d_pixels, info->width, info->width, d_hist);

    // Copy result back to CPU
    printf("Copying result to CPU\n");
    h_hist = (uint* )malloc(ACTIVE_CHANNELS * NUM_BINS * sizeof(uint));
    checkCudaErrors(cudaMemcpy(h_hist, d_hist, ACTIVE_CHANNELS * NUM_BINS * sizeof(uint), cudaMemcpyDeviceToHost));

    // Print results.
    print_histogram(h_hist);

    checkCudaErrors(cudaFree(d_pixels));
    checkCudaErrors(cudaFree(d_hist));
    free(h_pixels);
    free(h_hist);
	return 0;
}
