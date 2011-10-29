/*
 * Utilities.h
 *
 *  Created on: Oct 1, 2011
 *      Author: Dave
 */

#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <cuda.h>
#include <thrust/functional.h>

class Utilities
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
  struct NodeDistance
  {
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

  //----------------------------------------------------------------------------
  // Name:        NodeDistance1D
  // Description: A one-dimensional implementation of NodeDistance
  //----------------------------------------------------------------------------
  struct NodeDistance1D : public NodeDistance
  {
    __host__ __device__
    float operator()(const unsigned int idx1, const unsigned int idx2);
  };

  //----------------------------------------------------------------------------
  // Name:        NodeDistance2D
  // Description: A two-dimensional implementation of NodeDistance
  //----------------------------------------------------------------------------
  struct NodeDistance2D : public NodeDistance
  {
    NodeDistance2D(const unsigned int cols);

    __host__ __device__
    virtual float operator()(const unsigned int idx1, const unsigned int idx2);

  private:
    const unsigned int cols;
  };

  //----------------------------------------------------------------------------
  // Name:        NodeDistance3D
  // Description: A three-dimensional implementation of NodeDistance
  //----------------------------------------------------------------------------
  struct NodeDistance3D : public NodeDistance
  {
    NodeDistance3D(const unsigned int cols, const unsigned int sliceSize);

    __host__ __device__
    virtual float operator()(const unsigned int idx1, const unsigned int idx2);

  private:
    const unsigned int cols, sliceSize;
  };

  //----------------------------------------------------------------------------
  // Name:        SquaredDifference
  // Description: A generic functor to compute the squared difference
  //                between the two input values
  //----------------------------------------------------------------------------
  template <typename T>
  struct SquaredDifference : public thrust::binary_function<T, T, T>
  {
    __host__ __device__
    virtual T operator()(const T& x, const T&y) const
    { return (x - y) * (x - y); }
  };

  //----------------------------------------------------------------------------
  // Name:        AbsDifference
  // Description: A generic functor to compute the absolute value difference
  //                between the two input values
  //----------------------------------------------------------------------------
  template <typename T>
  struct AbsDifference : public thrust::binary_function<T, T, T>
  {
    __host__ __device__
    virtual T operator()(const T& x, const T&y) const
    { return (x > y ? (x - y) : (y - x)); }
  };

  //----------------------------------------------------------------------------
  // Name:        OperatorConstant
  // Description: A generic functor to compute the modulus between an input
  //                value and a constant
  //----------------------------------------------------------------------------
  template <typename T, typename Operator>
  struct OperatorConstant : public thrust::unary_function<T, T>
  {
    OperatorConstant(const T& constant) : constant(constant), op(Operator())
    { }

    __host__ __device__
    virtual T operator()(const T& x) const
    { return op(x, constant); }

  private:
    const T constant;
    const Operator op;
  };

  //----------------------------------------------------------------------------
  // Name:        ModConstant
  // Description: A generic functor to compute the modulus between an input
  //                value and a constant
  //----------------------------------------------------------------------------
  template <typename T>
  struct ModConstant : public OperatorConstant<T, thrust::modulus<T> >
  {
    ModConstant(const T& constant) :
      OperatorConstant<T, thrust::modulus<T> >(constant)
    { }
  };

  //----------------------------------------------------------------------------
  // Name:        DivideConstant
  // Description: A generic functor to compute the modulus between an input
  //                value and a constant
  //----------------------------------------------------------------------------
  template <typename T>
  struct DivideConstant : public OperatorConstant<T, thrust::divides<T> >
  {
    DivideConstant(const T& constant) :
      OperatorConstant<T, thrust::divides<T> >(constant)
    { }
  };

  // member methods
public:
  Utilities();
  virtual ~Utilities();

  // member variables
public:
  static const NodeDistance1D nodeDistance1d;
  static const NodeDistance2D nodeDistance2d;
  static const NodeDistance3D nodeDistance3d;
};

#endif /* UTILITIES_H_ */
