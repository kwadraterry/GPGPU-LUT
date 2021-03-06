#include "imageio.h"
#include <stdio.h>
#include <stdlib.h>

int read_png(char *file_name, png_infop *info, uchar4 **pixels) {
    char header[8];           // 8 is the maximum size that can be checked.
    int width, height;
    png_byte color_type;
    png_byte bit_depth;

    // Try to read a file; return error in case we failed.
    FILE *fp = fopen(file_name, "rb");
    if (fp == NULL) {
        return PNG_FAILURE;
    }
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING,
        NULL, NULL, NULL);

    if (!png) {
        return PNG_FAILURE;
    }

    *info = png_create_info_struct(png);
    if (!info) {
        return PNG_FAILURE;
    }

    if (setjmp(png_jmpbuf(png))) {
        return PNG_FAILURE;
    }

    png_init_io(png, fp);

    png_read_info(png, *info);

    width      = png_get_image_width(png, *info);
    height     = png_get_image_height(png, *info);
    color_type = png_get_color_type(png, *info);
    bit_depth  = png_get_bit_depth(png, *info);

    // Read any color_type into 8bit depth, RGBA format.
    // See http://www.libpng.org/pub/png/libpng-manual.txt

    if (bit_depth == 16) {
        png_set_strip_16(png);
    }

    if (color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_palette_to_rgb(png);
    }

    // PNG_COLOR_TYPE_GRAY_ALPHA is always 8 or 16 bit depth.
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
        png_set_expand_gray_1_2_4_to_8(png);
    }

    if (png_get_valid(png, *info, PNG_INFO_tRNS)) {
        png_set_tRNS_to_alpha(png);
    }

    // These color_type don't have an alpha channel then fill it with 0xff.
    if (color_type == PNG_COLOR_TYPE_RGB ||
        color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
    }

    if(color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_GRAY_ALPHA) {
        png_set_gray_to_rgb(png);
    }

    png_read_update_info(png, *info);

    // Allocate memory for the pixels.
    size_t bytes_per_row = png_get_rowbytes(png, *info);
    *pixels = (uchar4* )malloc(bytes_per_row * height);

    // libpng works with row pointers & byte pointers, but we wwant all bytes
    // stored continuously instead. We will 'trick' libpng into giving us all
    // byte data by iterating through its pointers, reading one row at a time.
    png_bytep row = (png_bytep) (*pixels);

    for (int i = 0; i < height; i++) {
        png_read_row(png, row, NULL);
        row += bytes_per_row;
    }

    // Finish reading.
    png_read_end(png, *info);

    fclose(fp);
    return PNG_SUCCESS;
}
