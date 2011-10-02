/*
 * Utilities.h
 *
 *  Created on: Oct 1, 2011
 *      Author: Dave
 */

#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <cuda.h>
#include "Kohonen.h"

class Utilities
{
  // member types
public:
  //----------------------------------------------------------------------------
  // Name:        NodeDistance1
  // Description: A one-dimensional implementation of NodeDistance
  //----------------------------------------------------------------------------
  class NodeDistance1 : public Kohonen::NodeDistance
  {
  public:
    __host__ __device__
    float operator()(const unsigned int idx1, const unsigned int idx2);
  };

  //----------------------------------------------------------------------------
  // Name:        NodeDistance2
  // Description: A two-dimensional implementation of NodeDistance
  //----------------------------------------------------------------------------
  class NodeDistance2 : public Kohonen::NodeDistance
  {
  public:
    NodeDistance2(const unsigned int cols);

    __host__ __device__
    virtual float operator()(const unsigned int idx1, const unsigned int idx2);

  private:
    const unsigned int cols;
  };

  //----------------------------------------------------------------------------
  // Name:        NodeDistance3
  // Description: A three-dimensional implementation of NodeDistance
  //----------------------------------------------------------------------------
  class NodeDistance3 : public Kohonen::NodeDistance
  {
  public:
    NodeDistance3(const unsigned int cols, const unsigned int sliceSize);

    __host__ __device__
    virtual float operator()(const unsigned int idx1, const unsigned int idx2);

  private:
    const unsigned int cols, sliceSize;
  };


public:
  Utilities();
  virtual ~Utilities();
};

#endif /* UTILITIES_H_ */
