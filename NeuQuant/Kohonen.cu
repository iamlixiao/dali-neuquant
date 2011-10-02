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
#include <thrust/device_vector.h>

//------------------------------------------------------------------------------
// Name:        Kohonen()
//------------------------------------------------------------------------------
Kohonen::Kohonen(const unsigned int numInputDimensions,
    const unsigned int numOutputDimensions,
    const unsigned int * const networkSize,
    NodeDistance* const nodeDistance) : numInputDimensions(numInputDimensions),
    numOutputDimensions(numOutputDimensions), networkSize(NULL),
    prodNetworkSize(1), nodeDistance(nodeDistance),
    ownNodeDistanceObject(false), randomInitialization(true)
{
  if (!nodeDistance)
  {
    switch(numOutputDimensions)
    {
    case 1:
      ownNodeDistanceObject = true;
      this->nodeDistance = new Utilities::NodeDistance1;
      break;
    case 2:
      ownNodeDistanceObject = true;
      this->nodeDistance = new Utilities::NodeDistance2(networkSize[1]);
      break;
    case 3:
      ownNodeDistanceObject = true;
      this->nodeDistance = new Utilities::NodeDistance3(networkSize[1],
          networkSize[0] * networkSize[1]);
      break;
    default:
      throw std::runtime_error(
          "Kohonen(): No node distance functor specified and "
          "numOutputDimensions > 3.");
    }
  }

  this->networkSize = new unsigned int[numOutputDimensions];
  memcpy(this->networkSize, networkSize,
      sizeof(unsigned int) * numOutputDimensions);

  for (unsigned int i = 0; i < numOutputDimensions; ++i)
    prodNetworkSize *= networkSize[i];

  network.resize(prodNetworkSize * numOutputDimensions);
}

Kohonen::~Kohonen()
{
  if (ownNodeDistanceObject && this->nodeDistance)
    delete nodeDistance;
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

//------------------------------------------------------------------------------
// Name:        train
//------------------------------------------------------------------------------
void Kohonen::train(const float * const points, const unsigned int numPoints,
    const unsigned int numIterators, const bool randomize)
{
  if (randomInitialization)
  {
    // Generate initialization on host
    thrust::host_vector<float> initialHost(
        prodNetworkSize * numOutputDimensions);
    thrust::generate(initialHost.begin(), initialHost.end(), rand);

    // Copy to device
    thrust::copy(initialHost.begin(), initialHost.end(), network.begin());
  }


}

