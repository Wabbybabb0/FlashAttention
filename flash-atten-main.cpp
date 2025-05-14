#include <torch/torch.h>
#include <iostream>

torch::Tensor my_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
// torch::Tensor forward_learn(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
// torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);


int main()
{
    const int batch_size = 2;
    const int n_head = 8;
    const int seq_len = 128; // 256
    const int head_embd = 64; // 128
    // 填充固定值便于检查
    auto q = torch::rand({batch_size, n_head, seq_len, head_embd}).cuda();
    auto k = torch::rand({batch_size, n_head, seq_len, head_embd}).cuda();
    auto v = torch::rand({batch_size, n_head, seq_len, head_embd}).cuda();
    // auto o1 = torch::rand({batch_size, n_head, seq_len, head_embd}).cuda();
    auto o2 = torch::rand({batch_size, n_head, seq_len, head_embd}).cuda();


    o2 = my_forward(q, k, v);
    // std::cout << "Output of my_forward:" << std::endl;
    // std::cout << o2[0][0][0]<< std::endl; 

    // o1 = forward_learn(q, k, v);
    // std::cout << "Output of forward:" << std::endl;
    // std::cout << o1[0][0][0] << std::endl; 

    return 0;
}