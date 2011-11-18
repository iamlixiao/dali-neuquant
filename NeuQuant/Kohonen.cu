/*
 * Kohonen.cu
 *
 *  Created on: Oct 29, 2011
 *      Author: David Bottisti
 */

#include "Kohonen.h"
#include <stdexcept>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

const unsigned int Kohonen::numInputDimensions = 3;
const unsigned int Kohonen::networkSize = 256;

// four primes near 500 - assume no image has a length so large
// that it is divisible by all four primes
const unsigned short Kohonen::primes[4] = {499, 491, 487, 503};

// Device Code
__shared__ float dist[Kohonen::networkSize];
__shared__ unsigned char bestIdx[Kohonen::networkSize];
__device__ float networkRed[Kohonen::networkSize];
__device__ float networkGreen[Kohonen::networkSize];
__device__ float networkBlue[Kohonen::networkSize];

__global__ void initializeNetwork(void)
{
  const unsigned int idx = threadIdx.x;
  networkRed[idx] = idx;
  networkGreen[idx] = idx;
  networkBlue[idx] = idx;
}

__global__ void trainOnPoint(const float * const points,
    const unsigned int step, const unsigned int pointsPerIteration)
{
  const unsigned int numInputPoints = gridDim.x * gridDim.y;
  const unsigned int tempPointIdx = blockIdx.y * gridDim.x + blockIdx.x;
  const unsigned int pointIdx = (tempPointIdx * step) % numInputPoints;
  const unsigned int networkIdx = threadIdx.x;
  const unsigned int iteration = tempPointIdx / pointsPerIteration;
  const float alpha = exp(-0.03 * iteration);
  const unsigned int radius = static_cast<unsigned int>(
      32.0 * exp(-0.0325 * iteration));

  __shared__ Kohonen::Color point;
  if (networkIdx == 0)
  {
    point.red = points[pointIdx];
    point.green = points[pointIdx + numInputPoints];
    point.blue = points[pointIdx + 2 * numInputPoints];
  }
  __syncthreads();

  // Compute the distance between the training point and the network
  dist[networkIdx] = fabs(point.red - networkRed[networkIdx]) +
      fabs(point.green - networkGreen[networkIdx]) +
      fabs(point.blue - networkBlue[networkIdx]);
  bestIdx[networkIdx] = networkIdx;

  // Wait for all the threads to compute their distance
  __syncthreads();

  // Reduce the distances
  for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
  {
    if (networkIdx < s && dist[networkIdx + s] < dist[networkIdx])
    {
      dist[networkIdx] = dist[networkIdx + s];
      bestIdx[networkIdx] = bestIdx[networkIdx + s];
    }

    __syncthreads();
  }
  const unsigned int winner = bestIdx[0];

  // Update the node based on the contents of the winning node, if it is within
  // the radius.
  const unsigned int dist = (winner < networkIdx ?
      networkIdx - winner : winner - networkIdx);
  if (dist < radius)
  {
    // 1) Compute alpha * rho based upon distance to winning index
    const float distOverRadius = static_cast<float>(dist) / radius;
    const float alphaRho = alpha * (1 - distOverRadius * distOverRadius);
    const float oneMinusAlphaRho = 1 - alphaRho;

    // 2) Compute the new node
    networkRed[networkIdx] = alphaRho * point.red +
        oneMinusAlphaRho * networkRed[networkIdx];
    networkGreen[networkIdx] = alphaRho * point.green +
        oneMinusAlphaRho * networkGreen[networkIdx];
    networkBlue[networkIdx] = alphaRho * point.blue +
        oneMinusAlphaRho * networkBlue[networkIdx];
  }
}

__global__ void mapPoints(float * const points)
{
  const unsigned int numInputPoints = gridDim.x * gridDim.y;
  const unsigned int pointIdx = blockIdx.y * gridDim.x + blockIdx.x;
  const unsigned int networkIdx = threadIdx.x;

  __shared__ Kohonen::Color point;
  if (networkIdx == 0)
  {
    point.red = points[pointIdx];
    point.green = points[pointIdx + numInputPoints];
    point.blue = points[pointIdx + 2 * numInputPoints];
  }
  __syncthreads();

  // Compute the distance between the image point and the network
  dist[networkIdx] = fabs(point.red - networkRed[networkIdx]) +
      fabs(point.green - networkGreen[networkIdx]) +
      fabs(point.blue - networkBlue[networkIdx]);
  bestIdx[networkIdx] = networkIdx;

  // Wait for all the threads in a block to compute their distance
  __syncthreads();

  // Reduce the distances by block
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (networkIdx < s && dist[networkIdx + s] < dist[networkIdx])
    {
      dist[networkIdx] = dist[networkIdx + s];
      bestIdx[networkIdx] = bestIdx[networkIdx + s];
    }

    __syncthreads();
  }

  // Copy the winning node to the output
  if (networkIdx == 0)
  {
    const unsigned int winningIdx = bestIdx[0];
    points[pointIdx] = networkRed[winningIdx];
    points[pointIdx + numInputPoints] = networkGreen[winningIdx];
    points[pointIdx + 2 * numInputPoints] = networkBlue[winningIdx];
  }
}

Kohonen::Kohonen(void)
{
  // Initialize the network
  initializeNetwork<<<1, networkSize>>>();
}

Kohonen::~Kohonen()
{
}

void Kohonen::train(const unsigned int width, const unsigned int height,
    float * const pointsHost)
{
  const unsigned int numInputPoints = width * height;
  const unsigned int pointsPerIteration = numInputPoints / 100;
  cudaError_t status;

  // Copy the image to the GPU
  float *points;
  cudaMalloc((void**)&points,
      sizeof(float) * numInputPoints * numInputDimensions);
  status = cudaGetLastError();
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("CUDA Error: ") +
        cudaGetErrorString(status));

  cudaMemcpy(points, pointsHost,
      sizeof(float) * numInputPoints * numInputDimensions,
      cudaMemcpyHostToDevice);
  status = cudaGetLastError();
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("CUDA Error: ") +
        cudaGetErrorString(status));

  // Pick a prime number close to 500 that is not a factor of the number of
  // pixels.  This is a step size to use for quasi-random training of the data
  unsigned int step = primes[3];
  for (unsigned int i = 0; i < 3; ++i)
  {
    if (numInputPoints % primes[i] != 0)
    {
      step = primes[i];
      break;
    }
  }

  dim3 gridDim(width, height, 1);

  // Train the network
  trainOnPoint<<<gridDim, networkSize>>>(points, step, pointsPerIteration);
  status = cudaGetLastError();
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("CUDA Error: ") +
        cudaGetErrorString(status));

  // Map the points to the output
  mapPoints<<<gridDim, networkSize>>>(points);
  status = cudaGetLastError();
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("CUDA Error: ") +
        cudaGetErrorString(status));

  // Copy the result back to the host
  cudaMemcpy(pointsHost, points,
      sizeof(float) * numInputPoints * numInputDimensions,
      cudaMemcpyDeviceToHost);
  status = cudaGetLastError();
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("CUDA Error: ") +
        cudaGetErrorString(status));

  cudaFree(points);
  status = cudaGetLastError();
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("CUDA Error: ") +
        cudaGetErrorString(status));
}
