
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
#include "histogram.h"

// Debug: Print pixel values.
void probe_image(PixelType* image, png_infop info) {
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


int main (int argc, char* argv[]) {
	PixelType* h_pixels;
    PixelType* d_pixels;
    unsigned int* d_hist;
    unsigned int* h_hist;

	png_infop info;

	if(argc < 2) {
		printf("Must include file name to process. `%s <file_name>`\n", argv[0]);
		return -1;
	}

	if(read_png(argv[1], &info, &h_pixels) == PNG_FAILURE) {
		printf("Error reading file (%s).\n", argv[1]);
		return -1;
	}

	size_t number_of_bytes = sizeof(uchar4) * info->width * info->height;

	printf("Image has height %lu and width %lu\n", info->width, info->width);


    // Copy data to GPU
    printf("...allocating GPU memory and copying input data\n\n");
    checkCudaErrors(cudaMalloc((void **)&d_pixels, number_of_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_hist, ACTIVE_CHANNELS * NUM_BINS * sizeof(uint)));
    checkCudaErrors(cudaMemcpy(d_pixels, h_pixels, number_of_bytes, cudaMemcpyHostToDevice));

    // Run histogram computing
    run_gmem_atomics(d_pixels, info->width, info->width, d_hist);
    printf("Done.\n");

    // Copy result back to CPU
    h_hist = (uint* )malloc(ACTIVE_CHANNELS * NUM_BINS * sizeof(uint));
    checkCudaErrors(cudaMemcpy(h_hist, d_hist, ACTIVE_CHANNELS * NUM_BINS * sizeof(uint), cudaMemcpyDeviceToHost));

    // Print results.
    printf("Histogram:\n");
    for (uint ch = 0; ch < ACTIVE_CHANNELS; ch++) {
    	printf("Channel %u:\n", ch + 1);
    	for (uint bin = 0; bin < NUM_BINS; bin++) {
        	printf("%5u ", h_hist[ch * NUM_BINS + bin]);
    	}
    	printf("\n");
    }

    checkCudaErrors(cudaFree(d_pixels));
    checkCudaErrors(cudaFree(d_hist));
    free(h_pixels);
    free(h_hist);
	return 0;
}
