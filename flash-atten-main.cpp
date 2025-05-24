#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#include "cudarand.h"

torch::Tensor forward_learn(torch::Tensor Q, torch::Tensor K, torch::Tensor V);


int main()
{
    // 参数设定
    const int batch_size = 2;
    const int n_head = 8;
    const int seq_len = 128;
    const int head_embd = 64;

    auto device = torch::kCUDA;

    // curand部分
    cudaStream_t stream = NULL;
    curandGenerator_t gen = NULL;
    curandRngType_t rng = CURAND_RNG_PSEUDO_MT19937;
    curandOrdering_t order = CURAND_ORDERING_PSEUDO_DEFAULT;

    const int n = batch_size * n_head * seq_len * head_embd;

    const unsigned long long seed_q = 1234ULL;
    const unsigned long long seed_k = 2345ULL;
    const unsigned long long seed_v = 3456ULL;

    const data_type mean = 1.0f;
    const data_type stddev = 2.0f;

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::vector<data_type> hq_data(n, 0);
    std::vector<data_type> hk_data(n, 0);
    std::vector<data_type> hv_data(n, 0);
    std::vector<int64_t> shape = {batch_size, n_head, seq_len, head_embd};

    run_on_host(n, mean, stddev, seed_q, order, rng, stream, gen, hq_data);
    run_on_host(n, mean, stddev, seed_k, order, rng, stream, gen, hk_data);
    run_on_host(n, mean, stddev, seed_v, order, rng, stream, gen, hv_data);
    for(int i = 0; i < 12; i++){
        std::cout << hq_data[i] << std::endl;
    }

    // 从vector到Tensor
    torch::Tensor v2t_q = torch::from_blob(hq_data.data(), shape, torch::kFloat).clone();
    torch::Tensor v2t_k = torch::from_blob(hk_data.data(), shape, torch::kFloat).clone();
    torch::Tensor v2t_v = torch::from_blob(hv_data.data(), shape, torch::kFloat).clone();

    torch::Tensor q = v2t_q.to(torch::kCUDA);
    torch::Tensor k = v2t_k.to(torch::kCUDA);
    torch::Tensor v = v2t_v.to(torch::kCUDA);

    // 1.用时检测
    // 预热
    for (int i = 0; i < 10; ++i) {
        auto _ = forward_learn(q, k, v);
    }
    torch::cuda::synchronize();

    // 计时开始
    auto start = std::chrono::high_resolution_clock::now();
    auto out = forward_learn(q, k, v);
    torch::cuda::synchronize();

    // 计时结束
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "my_forward执行时间: " << duration.count() << " ms" << std::endl;

    // 2.正确验证(打印前256个结果)
    auto my_out_flat = out.view(-1);
    for(int i = 0; i < 256; i++){
        std::cout<< "my_learn[" << i << "]:"<< my_out_flat[i].item<float>() << std::endl;
    }
    return 0;
}