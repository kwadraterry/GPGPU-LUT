# Color Histogram GPU/CPU implementation.

## Dependencies
Requires: 
* CUDA (not sure what version is enough, but developing and testing was made with CUDA 8.0) with SDK.
* libpng (at least 1.2) for reading image files.

## Build
(Requires CMake of at least 2.8)
1. ``` mkdir build & cd build```
2. ``` cmake .. ```
3. ``` make ```

## Run
Usage: 

```./ColorHistogram file.png [options]```

Options: 

* ```-t N```, ```--times N```  - Run both GPU and CPU version N times; output averaged time and speed. Default = 1. 
* ```-h```, ```--help```     - Print help message. 

