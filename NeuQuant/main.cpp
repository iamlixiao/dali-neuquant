/*
 * main.cpp
 *
 *  Created on: Sep 25, 2011
 *      Author: Dave
 */

#include "NEUQUANT.h"
#include "CImg.h"
#include <time.h>
#include <unistd.h>

int main(const unsigned int argc, const char * const * const argv)
{
  clock_t elapsedTime = 0, startTime;
  cimg_library::cimg::exception_mode(1);

  // Load image
  cimg_library::CImg<unsigned char> imgRGBSlices;

  for (unsigned int j = 1; j < argc; ++j)
  {
    try
    {
      printf("Trying file: %s\n", argv[j]);
      imgRGBSlices.load_jpeg(argv[j]);
      if (imgRGBSlices.spectrum() != 3)
      {
        printf("  Error: File not color.  Skipping\n");
        continue;
      }

      unsigned char *imgBGR = new unsigned char[imgRGBSlices.size()];

      startTime = clock();

      // Reshape the image into BGR order
      unsigned char *red = imgRGBSlices.data(0, 0, 0, 0), *green = imgRGBSlices.data(0, 0, 0, 1),
          *blue = imgRGBSlices.data(0, 0, 0, 2);
      unsigned char *out = imgBGR;
      const unsigned int size = imgRGBSlices.width() * imgRGBSlices.height();
      for (unsigned int i = 0; i < size; ++i, ++red, ++green, ++blue)
      {
        out[0] = *blue;
        out[1] = *green;
        out[2] = *red;
        out += 3;
      }

      // Initialize neuquant
      initnet(imgBGR, 3*size, 1);

      // Perform training
      learn();
      unbiasnet();

      // Create output image (overwrite imgRGBSlices)
      red = imgRGBSlices.data(0, 0, 0, 0);
      green = imgRGBSlices.data(0, 0, 0, 1);
      blue = imgRGBSlices.data(0, 0, 0, 2);
      for (unsigned int i = 0; i < size; ++i, ++red, ++green, ++blue)
      {
        unsigned char index = inxsearch(*blue, *green, *red);
        *red = getNetwork(index, 2);
        *green = getNetwork(index, 1);
        *blue = getNetwork(index, 0);
      }
      elapsedTime += (clock() - startTime);
      printf("  Elapsed time: %f sec, Average time: %f sec\n", static_cast<double>(elapsedTime) / CLOCKS_PER_SEC,
          static_cast<double>(elapsedTime / j) / CLOCKS_PER_SEC);

#if cimg_display != 0
      imgRGBSlices.display();
#endif

      delete [] imgBGR;
    }
    catch (std::exception &e)
    {
      unlink(argv[j]);
      printf("  Error: File not a jpeg.  Skipping.\n");
    }
  }

  printf("Total time: %f seconds.\n", static_cast<double>(elapsedTime) / CLOCKS_PER_SEC);

  return 0;
}

