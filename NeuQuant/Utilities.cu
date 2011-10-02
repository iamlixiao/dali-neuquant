/*
 * Utilities.cpp
 *
 *  Created on: Oct 1, 2011
 *      Author: Dave
 */

#include "Utilities.h"

//------------------------------------------------------------------------------
// Name:        NodeDistance1::operator()
//------------------------------------------------------------------------------
float Utilities::NodeDistance1::operator ()(const unsigned int idx1,
    const unsigned int idx2)
{
  return static_cast<float>(idx1 < idx2 ? idx2 - idx1 : idx1 - idx2);
}

//------------------------------------------------------------------------------
// Name:        NodeDistance2::NodeDistance2()
//------------------------------------------------------------------------------
Utilities::NodeDistance2::NodeDistance2(const unsigned int cols) : cols(cols)
{ }

//------------------------------------------------------------------------------
// Name:        NodeDistance2::operator()
//------------------------------------------------------------------------------
float Utilities::NodeDistance2::operator()(const unsigned int idx1,
    const unsigned int idx2)
{
  const unsigned int row1 = idx1 / cols, col1 = idx1 % cols;
  const unsigned int row2 = idx2 / cols, col2 = idx2 % cols;

  return sqrt(static_cast<float>((row1 - row2) * (row1 - row2) +
      (col1 - col2) * (col1 - col2)));
}

//------------------------------------------------------------------------------
// Name:        NodeDistance3::NodeDistance3()
//------------------------------------------------------------------------------
Utilities::NodeDistance3::NodeDistance3(const unsigned int cols,
    const unsigned int sliceSize) : cols(cols), sliceSize(sliceSize)
{ }

//------------------------------------------------------------------------------
// Name:        NodeDistance3::operator()
//------------------------------------------------------------------------------
float Utilities::NodeDistance3::operator()(const unsigned int idx1,
    const unsigned int idx2)
{
  const unsigned int slice1 = idx1 / sliceSize,
      idx1InSlice = idx1 % sliceSize, row1 = idx1InSlice / cols,
      col1 = idx1InSlice % cols;
  const unsigned int slice2 = idx2 / sliceSize,
      idx2InSlice = idx2 % sliceSize, row2 = idx2InSlice / cols,
      col2 = idx2InSlice % cols;
  return sqrt(static_cast<float>((slice1 - slice2) * (slice1 - slice2) +
      (row1 - row2) * (row1 - row2) +
      (col1 - col2) * (col1 - col2)));
}

