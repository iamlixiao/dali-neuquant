/*
 * Kohonen.h
 *
 *  Created on: Oct 1, 2011
 *      Author: Dave
 */

#ifndef KOHONEN_H_
#define KOHONEN_H_

#include <math.h>
#include <stdlib.h>
#include <thrust/device_vector.h>

//------------------------------------------------------------------------------
// Class:       Kohonen
// Description: A class to train an arbitrary dimensional Kohonen Neural
//              Network given training points
//------------------------------------------------------------------------------
class Kohonen
{
  // member types
public:
  //----------------------------------------------------------------------------
  // Name:        NodeDistance
  // Description: A functor whose operator()(unsigned int, unsigned int) method
  //                returns the distance between two nodes whose indices are
  //                provided.  The user may override the method in this class
  //                to override the default behavior.  Specific implementations
  //                for 1, 2 and 3 dimensional networks is provided, and are
  //                used by default when possible
  //----------------------------------------------------------------------------
  class NodeDistance
  {
  public:
    //--------------------------------------------------------------------------
    // Name:        operator()
    // Description: Returns the distance between the two network nodes given
    // Parameters:  idx1 - The index of the first node
    //              idx2 - The index of the second node
    // Returns:     The distance between nodes at idx1 and idx2
    // Notes:       None
    //--------------------------------------------------------------------------
    virtual float operator()(const unsigned int idx1,
        const unsigned int idx2) = 0;
  };

  // member methods
public:
  //----------------------------------------------------------------------------
  // Name:        Kohonen
  // Description: Constructor
  // Parameters:  numInputDimensions - The number of dimensions of the input
  //                training data
  //              numOutputDimensions - The number of dimensions of the network.
  //                This is usally less than numInputDimensions
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
      NodeDistance* const nodeDistance = NULL);

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
  void train(const float * const points, const unsigned int numPoints,
      const unsigned int numIterators, const bool randomize = false);

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
  NodeDistance* nodeDistance;
  bool ownNodeDistanceObject;

  thrust::device_vector<float> network;
  bool randomInitialization;
};

#endif /* KOHONEN_H_ */
