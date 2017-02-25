```CUDA
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
 ```
 
Gather the information from a part of the image to the partial hisogram.
Part of the image is defined by block and grid size ("part of image" is not a continious patch here, rather every n-th pixel).
The partial histogram is stored in global memory defined by "out". Every block has a group number (g) assigned to it, 
and it treats part of global memory (from out + g * NUM_PARTS and every nt bits, where nt is the size (x*y) of the grid) 
as its own, storing the histogram of its part of the image here.

What happens in the kernel is:

1. the blocks' own bits of global memory get initialized with zeros.

2. Every corresponding (assigned to this block) pixel get visited, and: 

   a. Pixel is decoded into several (ACTIVE_CHANNELS) colors.
   
   b. These values are binned to the desired number of bins (NUM_BINS).
   
   c. Corresponding values in a partial histogram (part of global memory) get updated.
 
 Parts a and b are implemented in *DecodePixel*
 
```CUDA
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
```

This kernel gathers partial histograms from global memory (in) into output histogram.

Partial histograms are expected to be stored in a part of global memory of n * ACTIVE_CHANNELS * NUM_BINS * sizeof(uint) size. 
It is expected that every nth bit belongs to the same partial histogram (the k-th histogram stored in k, n + k, 2 * n + k, ...
elements).

Every block running the kernel has a particular bin on the output histogram it is responsible of. It gathers information
about this bin from every partial histogram, and sums it up.


```CUDA
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
```
This is a CPU-wrapper of the histogram computation. It assigns the blocks, allocates the memory for partial histograms, 
computes them using *histogram_gmem_atomics* and accumulates with *histogram_gmem_accum*.

```CUDA
__global__ void histogram_gmem_atomics1(
        const PixelType *in,
        int width,
        int height,
        unsigned int *out,
        int dev_id,
        int dev_count)
{
```

Same as *histogram_gmem_atomics*, but with extra parallelization based on channel (meant for multi-GPU, so that every device 
has specific channel(s) assigned to it).

```CUDA
void run_multigpu(
        PixelType *d_image,
        int width,
        int height,
        unsigned int *d_hist,
        int device_id,
        int device_count)
{
```

Version of *run_gmem_atomics* for multiple GPUs. The only part that differs is histogram_gmem_atomics1, gathering only
the specified channels. Accumulating is still done over the whole global memory array.
