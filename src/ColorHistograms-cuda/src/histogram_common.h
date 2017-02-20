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

#ifndef HISTOGRAM_COMMON_H_
#define HISTOGRAM_COMMON_H_

// This header contains constants that are controlling the histogram calculation.
// They affect both CPU and GPU versions.


// These are fine to experiment with. They actually would look better as parameters,
//      but I am too lazy for this.
#define NUM_BINS 16              // Number of bins.
#define ACTIVE_CHANNELS 4        // Number of channels for which the histogram is computed.
                                 //    Note: PNG has 4 channels (R, G, B, Alpha).
                                 //    So, to compute histograms only for RGB, without
                                 //    alpha-channel, set this to 3.


// These are for inner use. Nothing stops you from modyfying these, too,
//    but that would probably break something.
#define PixelType uchar4         // Type used for pixel data.
#define K_BIN (256 / NUM_BINS)	 // Number of colors stored in one bin.
#define NUM_PARTS NUM_BINS * ACTIVE_CHANNELS // Size of partial histogram.



#endif /* HISTOGRAM_COMMON_H_ */
