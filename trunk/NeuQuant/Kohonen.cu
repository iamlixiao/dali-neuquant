/*
 * Kohonen.cpp
 *
 *  Created on: Oct 1, 2011
 *      Author: Dave
 */

#include "Kohonen.h"
#include "Utilities.h"
#include <stdexcept>
#include <thrust/copy.h>
#include <thrust/host_vector.h>

//------------------------------------------------------------------------------
// Name:        Kohonen()
//------------------------------------------------------------------------------
Kohonen::Kohonen(const unsigned int numInputDimensions,
    const unsigned int numOutputDimensions,
    const unsigned int * const networkSize,
    const Utilities::NodeDistance& nodeDistance) :
    numInputDimensions(numInputDimensions),
    numOutputDimensions(numOutputDimensions), networkSize(NULL),
    prodNetworkSize(1), nodeDistance(nodeDistance), randomInitialization(true)
{
  this->networkSize = new unsigned int[numOutputDimensions];
  memcpy(this->networkSize, networkSize,
      sizeof(unsigned int) * numOutputDimensions);

  for (unsigned int i = 0; i < numOutputDimensions; ++i)
    prodNetworkSize *= networkSize[i];

  network.resize(prodNetworkSize * numOutputDimensions);
}

Kohonen::~Kohonen()
{
  if (networkSize)
    delete [] networkSize;
}

//------------------------------------------------------------------------------
// Name:        initialize
//------------------------------------------------------------------------------
void Kohonen::initialize(const float * const nodes)
{
  randomInitialization = false;

  thrust::copy(nodes, nodes + prodNetworkSize * numOutputDimensions,
      network.begin());
}

