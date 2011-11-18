/*
 * Kohonen.h
 *
 *  Created on: Oct 29, 2011
 *      Author: David Bottisti
 */

#ifndef KOHONEN_H_
#define KOHONEN_H_

#include <math.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>

//------------------------------------------------------------------------------
// Class:       Kohonen
// Description: A class to train a 1-dimensions Kohonen Neural Network given
//              3-dimensional training points
//------------------------------------------------------------------------------
class Kohonen
{
  // member types
public:
  struct Color
  {
    float red;
    float green;
    float blue;
  };

  // member methods
public:
  Kohonen(void);

  void train(const unsigned int width, const unsigned int height,
      float * const pointsHost);

  virtual ~Kohonen();

  // member variables
public:
  static const unsigned int numInputDimensions;
  static const unsigned int networkSize;
  static const unsigned short primes[4];
};

#endif /* KOHONEN_H_ */
