#include <cuda_runtime.h>
#include <curand.h>

#include "./utils/curand_utils.h"
using data_type = float;
void run_on_device(const int &n, const data_type &mean, const data_type &stddev,
                   const unsigned long long &seed,
                   const curandOrdering_t &order, const curandRngType_t &rng,
                   const cudaStream_t &stream, curandGenerator_t &gen,
                   std::vector<data_type> &h_data) {

  data_type *d_data = nullptr;

  /* C data to device */
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_data),
                        sizeof(data_type) * h_data.size()));

  /* Create pseudo-random number generator */
  CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937));

  /* Set cuRAND to stream */
  CURAND_CHECK(curandSetStream(gen, stream));

  /* Set ordering */
  CURAND_CHECK(curandSetGeneratorOrdering(gen, order));

  /* Set seed */
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));

  /* Generate n floats on device */
  CURAND_CHECK(
      curandGenerateLogNormal(gen, d_data, h_data.size(), mean, stddev));

  /* Copy data to host */
  CUDA_CHECK(cudaMemcpyAsync(h_data.data(), d_data,
                             sizeof(data_type) * h_data.size(),
                             cudaMemcpyDeviceToHost, stream));

  /* Sync stream */
  CUDA_CHECK(cudaStreamSynchronize(stream));

  /* Cleanup */
  CUDA_CHECK(cudaFree(d_data));
}

void run_on_host(const int &n, const data_type &mean, const data_type &stddev,
                 const unsigned long long &seed, const curandOrdering_t &order,
                 const curandRngType_t &rng, const cudaStream_t &stream,
                 curandGenerator_t &gen, std::vector<data_type> &h_data) {

  /* Create pseudo-random number generator */
  CURAND_CHECK(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_MT19937));

  /* Set cuRAND to stream */
  CURAND_CHECK(curandSetStream(gen, stream));

  /* Set ordering */
  CURAND_CHECK(curandSetGeneratorOrdering(gen, order));

  /* Set seed */
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));

  /* Generate n floats on host */
  CURAND_CHECK(
      curandGenerateLogNormal(gen, h_data.data(), h_data.size(), mean, stddev));

  /* Cleanup */
  CURAND_CHECK(curandDestroyGenerator(gen));
}
