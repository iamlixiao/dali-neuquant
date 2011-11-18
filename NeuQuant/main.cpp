/*
 * main.cpp
 *
 *  Created on: Sep 25, 2011
 *      Author: David Bottisti
 */

#include "NEUQUANT.h"
#include "Kohonen.h"
#include "CImg.h"

#include <cuda.h>
#include <time.h>
#include <unistd.h>

int main(const int argc, const char * const * const argv)
{
  // Note:  Change sequential to 'true' to run the GPU version
  const unsigned int sequential = false;

  double elapsedTime = 0, thisTime = 0, startTime;
  struct timespec tp;
  cimg_library::cimg::exception_mode(1);

  // Load image
  cimg_library::CImg<float> imgRGBSlices;

  try
  {
    imgRGBSlices.load_jpeg(argv[1]);
    if (imgRGBSlices.spectrum() != 3)
      return 1;
    const unsigned int size = imgRGBSlices.width() * imgRGBSlices.height();

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tp);
    startTime = tp.tv_sec + tp.tv_nsec * 0.000000001;

    if (sequential)
    {
      unsigned char *imgBGR = new unsigned char[imgRGBSlices.size()];
      // Reshape the image into BGR order
      float *red = imgRGBSlices.data(0, 0, 0, 0),
          *green = imgRGBSlices.data(0, 0, 0, 1),
          *blue = imgRGBSlices.data(0, 0, 0, 2);
      unsigned char *out = imgBGR;
      for (unsigned int i = 0; i < size; ++i, ++red, ++green, ++blue)
      {
        out[0] = static_cast<unsigned int>(*blue);
        out[1] = static_cast<unsigned int>(*green);
        out[2] = static_cast<unsigned int>(*red);
        out += 3;
      }

      red = imgRGBSlices.data(0, 0, 0, 0);
      green = imgRGBSlices.data(0, 0, 0, 1);
      blue = imgRGBSlices.data(0, 0, 0, 2);

      // Initialize neuquant
      initnet(imgBGR, 3*size, 1);

      // Perform training
      learn();
      unbiasnet();

        // Create output image (overwrite imgRGBSlices)
      for (unsigned int i = 0; i < size; ++i, ++red, ++green, ++blue)
      {
        unsigned char index = inxsearch(*blue, *green, *red);
        *red = getNetwork(index, 2);
        *green = getNetwork(index, 1);
        *blue = getNetwork(index, 0);
      }
      delete [] imgBGR;
    }
    else
    {
      Kohonen kohonen;
      kohonen.train(imgRGBSlices.width(), imgRGBSlices.height(),
          imgRGBSlices.data());
    }

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tp);
    thisTime = (tp.tv_sec + tp.tv_nsec * 0.000000001) - startTime;
    elapsedTime += thisTime;

    printf("%d  %f\n", size, thisTime);

  }
  catch (std::exception &e)
  {
      printf("  Error: File not a jpeg.  Skipping.\n");
  }

  return 0;
}

