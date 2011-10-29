/*
 * Kohonen.h
 *
 *  Created on: Oct 1, 2011
 *      Author: Dave
 */

#ifndef KOHONEN_H_
#define KOHONEN_H_

#include "Utilities.h"

#include <math.h>
#include <cstdlib>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/discard_iterator.h>

//------------------------------------------------------------------------------
// Class:       Kohonen
// Description: A class to train an arbitrary dimensional Kohonen Neural
//              Network given training points
//------------------------------------------------------------------------------
class Kohonen
{
  // member methods
public:
  //----------------------------------------------------------------------------
  // Name:        Kohonen
  // Description: Constructor
  // Parameters:  numInputDimensions - The number of dimensions of the input
  //                training data
  //              numOutputDimensions - The number of dimensions of the network.
  //                This is usually less than numInputDimensions
  //              networkSize - An array of numOutputDimensions elements, each
  //                of which indicates the size of the network space in the
  //                corresponding dimension.  I.e., the i'th item in
  //                networkSize is the size of the i'th dimension of the
  //                network space
  //              nodeDistance - A pointer to a functor used to compute the
  //                distances between nodes.  If specified, it is used instead
  //                of the default.  If not specified but numOutputDimensions
  //                is greater than 3, a runtime_error is thrown.
  //                (Default: NULL, i.e., use default)
  // Returns:     N/A
  // Notes:       None
  //----------------------------------------------------------------------------
  Kohonen(const unsigned int numInputDimensions,
      const unsigned int numOutputDimensions,
      const unsigned int * const networkSize,
      const Utilities::NodeDistance& nodeDistance = Utilities::nodeDistance1d);

  //----------------------------------------------------------------------------
  // Name:        ~Kohonen
  // Description: Destructor
  // Parameters:  N/A
  // Returns:     N/A
  // Notes:       None
  //----------------------------------------------------------------------------
  virtual ~Kohonen();

  //----------------------------------------------------------------------------
  // Name:        initialize
  // Description: Initializes the neural network with the given nodes
  // Parameters:  nodes - An array of product(networkSize) rows and
  //                numOutputDimension columns, in row-major order.  These are
  //                the nodes use to initialize the network.
  // Returns:     N/A
  // Notes:       - Initialization is optional.  If not performed, the network
  //                is initialized to random points.
  //              - If performed, and if relying on the built in node-distance
  //                functions, the nodes should be specified in extended
  //                row-major order.  That is, a linear scan of the nodes
  //                should scan across the first dimension (rows) followed by
  //                the second dimension (columns), then the third (slice), etc.
  //----------------------------------------------------------------------------
  void initialize(const float * const nodes);

  //----------------------------------------------------------------------------
  // Name:        train
  // Description: Trains the neural network on the points.  Either sequential or
  //                random order training may be selected
  // Parameters:  points - An array of numPoints rows and numInputDimensions
  //                columns, in row-major order.  These are the points to train.
  //              numPoints - The number of points
  //              numIterations - The number of iterations of training to
  //                perform
  //              randomize - True if the points should be shuffled randomly
  //                before each iteration, false otherwise. (Default: FALSE)
  // Returns:     N/A
  // Notes:       None
  //----------------------------------------------------------------------------
  template <typename WeightDifferenceOperator,
    typename WeightReductionOperator>
  void train(const float * const points, const unsigned int numPoints,
      const unsigned int numIterations, const bool randomize = false)
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

    // Copy the points to the device
    thrust::device_vector<float> pointsDevice(points,
        points + numPoints * numInputDimensions),
        differences(numPoints * numInputDimensions), distances(numPoints);

    for (unsigned int i = 0; i < numIterations; ++i)
    {
      // Loop through all of the points
      thrust::device_vector<float>::const_iterator currentPoint =
          pointsDevice.begin();
      for (unsigned int j = 0; j < numPoints;
          ++j, currentPoint += numInputDimensions)
      {
        // Compute the point-by-point difference using WeightDifferenceOperator
        thrust::transform(pointsDevice.begin(), pointsDevice.end(),
            thrust::make_permutation_iterator(
                currentPoint,
                thrust::make_transform_iterator(
                    thrust::make_counting_iterator(0u),
                    Utilities::ModConstant<unsigned int>(numInputDimensions))),
            differences.begin(),
            WeightDifferenceOperator());

        // Reduce each row into a single distance using WeightReductionOperator
        thrust::reduce_by_key(
            thrust::make_transform_iterator(thrust::make_counting_iterator(0u),
                Utilities::DivideConstant<unsigned int>(numInputDimensions)),
            thrust::make_transform_iterator(thrust::make_counting_iterator(0u),
                Utilities::DivideConstant<unsigned int>(numInputDimensions)) +
                differences.size(),
            differences.begin(),
            thrust::make_discard_iterator(),
            distances.begin(),
            thrust::equal_to<unsigned int>(),
            WeightReductionOperator());

        // Find the winner (i.e., node with closest weight)
        thrust::tuple<float, unsigned int> winner = thrust::min_element(
            thrust::make_zip_iterator(
                thrust::make_tuple(distances.begin(),
                    thrust::make_counting_iterator(0u))),
            thrust::make_zip_iterator(
                thrust::make_tuple(distances.begin(),
                    thrust::make_counting_iterator(0u))) + distances.size());
        const unsigned int winnerIdx = thrust::get<1>(winner);

        // Update the weights based upon the node distance to the winner
      }
    }

  }

  //----------------------------------------------------------------------------
  // Name:        getPoints
  // Description: Determine the mapping from the input points to points in the
  //                neural network
  // Parameters:  points - An array of numPoints rows and numInputDimensions
  //                columns, in row-major order.  The input points to map.
  //              mapped [output] - A pre-allocated array the same size as
  //                points.  This method populates mapped with the
  //                weights associated with the closest node to each point in
  //                points.  In-place mapping is supported when mapped == NULL
  //                (Default: mapped = NULL, i.e., in-place)
  // Returns:     N/A
  // Notes:       None
  //----------------------------------------------------------------------------
  void getPoints(const float * const points, float * const mapped = NULL);

  // member variables
private:
  const unsigned int numInputDimensions;
  const unsigned int numOutputDimensions;
  unsigned int *networkSize;
  unsigned int prodNetworkSize;
  const Utilities::NodeDistance &nodeDistance;

  thrust::device_vector<float> network;
  bool randomInitialization;
};

#endif /* KOHONEN_H_ */
