#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cstdio>
#include <memory>
#include <string>
#include <sstream>
#include <array>

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "cudarand.h"

// Attention 实现
torch::Tensor attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, int head_embd) {
    auto scores = torch::matmul(q, k.transpose(-2, -1));
    // 测试q[1][i]和k[0][i]相乘后的值================================
    // float sum = 0.0;
    // for(int i = 0; i < 64; i++){
    //     std::cout << "q[" << i << "]:" << q[0][0][1][i].item<float>() << " × " << "k[" << i << "]:" << k[0][0][96][i].item<float>() << "=" << q[0][0][1][i].item<float>()*k[0][0][96][i].item<float>() << std::endl;
    //     sum += q[0][0][1][i].item<float>() * k[0][0][96][i].item<float>();
    // }
    // std::cout << "sum = " << sum << std::endl;
    // ================================

    auto weights = torch::softmax(scores / head_embd, -1);
    // for(int i = 0; i < 128; i++){
    //     // std::cout << "scores[" << i << "]:" << scores[0][0][1][i].item<float>() << std::endl;
    //     float val = weights[0][0][1][i].item<float>();
    //     printf("weights[%d]: %.7f\n", i, val);
    //     // std::cout << "weights[" << i << "]:" << weights[0][0][1][i].item<float>() << std::endl;
    // }

    auto output = torch::matmul(weights, v);
    return output;
}

int main() {
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

    // 从vector到Tensor
    torch::Tensor v2t_q = torch::from_blob(hq_data.data(), shape, torch::kFloat).clone();
    torch::Tensor v2t_k = torch::from_blob(hk_data.data(), shape, torch::kFloat).clone();
    torch::Tensor v2t_v = torch::from_blob(hv_data.data(), shape, torch::kFloat).clone();

    // 1. 用时检测
    // 预热
    for (int i = 0; i < 10; ++i) {
        auto _ = attention(v2t_q, v2t_k, v2t_v, head_embd);
    }
    cudaDeviceSynchronize(); // 保证预热完成

    // 计时开始
    auto start = std::chrono::high_resolution_clock::now();

    auto out = attention(v2t_q, v2t_k, v2t_v, head_embd);

    // 计时结束
    auto end = std::chrono::high_resolution_clock::now();

    // 计算耗时
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Attention执行时间: " << duration.count() << " us" << std::endl;

    // 2.正确验证
    auto out_flat = out.view(-1);
    for(int i = 0; i < 256; i++){
        std::cout<< "out_flat[" << i << "]:"<< out_flat[i].item<float>() << std::endl;
    }
    std::cout << "over!" << std::endl;
    return 0;
}