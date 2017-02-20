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

```
./ColorHistogram file.png [options]
```

Options: 

* ```-t N```, ```--times N``` - Run each configuration N times; output averaged time and speed. Default = 1.

* ```-C```, ```-c```, ```--cpu``` - Run on CPU                   

* ```-G```, ```-g```, ```--gpu``` - Run on GPU                   

* ```-M```, ```-m```, ```--mgpu``` - Run on multiple GPUs         

  If none of ```C```, ```G``` or ```M``` flags are specified, defaults to all (run every configuration).
                               
* ```-V```, ```-v```, ```--verbose``` - Output activity information. 

* ```-Q```, ```-q```, ```--quiet``` - Dont't print the histogram.  

* ```-h```, ```--help``` - Print usage information.



