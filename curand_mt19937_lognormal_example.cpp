/*
 * This program uses the host CURAND API to generate 100
 * pseudorandom floats.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "cudarand.h"

int main(int argc, char *argv[]) {

  cudaStream_t stream = NULL;
  curandGenerator_t gen = NULL;
  curandRngType_t rng = CURAND_RNG_PSEUDO_MT19937;
  curandOrdering_t order = CURAND_ORDERING_PSEUDO_DEFAULT;

  const int n = 12;

  const unsigned long long seed = 1234ULL;

  const data_type mean = 1.0f;
  const data_type stddev = 2.0f;

  /* Create stream */
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  /* Allocate n floats on host */
  std::vector<data_type> h_data(n, 0);

  run_on_host(n, mean, stddev, seed, order, rng, stream, gen, h_data);

  printf("Host\n");
  print_vector(h_data);
  printf("=====\n");

  run_on_device(n, mean, stddev, seed, order, rng, stream, gen, h_data);

  printf("Device\n");
  print_vector(h_data);
  printf("=====\n");

  /* Cleanup */
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaDeviceReset());

  return EXIT_SUCCESS;
}