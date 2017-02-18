
// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Helper functions
#include <helper_functions.h>  // CUDA SDK Helper functions
#include <helper_cuda.h>       // CUDA device initialization helper functions
#include <helper_cuda_gl.h>    // CUDA device + OpenGL initialization functions

#include <stdlib.h>
#include <stdio.h>
#include <png.h>

#include "imageio.h"


int main (int argc, char* argv[]) {
	uchar4 *host_in;

	png_infop info;

	if(argc < 2) {
		printf("Must include file name to process. `%s <file_name>`\n", argv[0]);
		return -1;
	}

	if(read_png(argv[1], &info, &host_in) == PNG_FAILURE) {
		printf("Error reading file (%s).\n", argv[1]);
		return -1;
	}

	size_t number_of_bytes = sizeof(uchar4) * info->width * info->height;

	printf("Image has height %lu and width %lu\n", info->height, info->width);
	return 0;
}
