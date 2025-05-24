#include<torch/types.h>
#include<cuda.h>
#include<cuda_runtime.h>

__global__ void forward_learn_kernel(const float *Q, const float *K, const float *V,
                                     const int N, const int d, const int Br, const int Bc, const int Tr, const int Tc, const float softmax_scale,
                                     float *m, float *l, float *O){
    int tid = threadIdx.x;
    int bid = gridDim.y * blockIdx.x + blockIdx.y;
    int thread_per_block = blockDim.x * blockDim.y;
    
    // Q: Br * d, K: Bc * d, V: Bc * d, S: Br * Bc
    extern __shared__ float sram[];
    float *Q_s = sram;
    int offset = Br * d;
    float *K_s = &sram[offset];
    offset += Bc * d;
    float *V_s = &sram[offset];
    offset += Bc * d;
    float *S_s = &sram[offset];
    
    // 定位获取输入的数据
    const float *Q_start = Q + bid * N * d;
    const float *K_start = K + bid * N * d;
    const float *V_start = V + bid * N * d;
    float *O_start = O + bid * N * d;
    float *m_start = m + bid * N;
    float *l_start = l + bid * N;
    
    for(int j = 0; j < Tc; j++)
    {
        // 载入K、V到sram
        for(int k = 0; k < Bc * d; k += thread_per_block){
            K_s[k + tid] = K_start[j * Bc * d + k + tid];
            V_s[k + tid] = V_start[j * Bc * d + k + tid];
        }
        __syncthreads();

        for(int i = 0; i < Tr; i++){
            // 加载Q到sram
            for(int k = 0; k < Br * d; k += thread_per_block){
                Q_s[k + tid] = Q_start[i * Br * d + k + tid];
            }
            __syncthreads();

            float row_m = -INFINITY;
            for(int y = 0; y < Bc; y++){
                float sum = 0;
                for(int x = 0; x < d; x++){
                    sum += Q_s[tid * d + x] * K_s[y * d + x];
                }
                sum *= softmax_scale;
                S_s[tid * Bc + y] = sum;

                if(sum > row_m){
                    row_m = sum;
                }
            }
            float row_l = 0;
            for(int y = 0; y < Bc; y++){
                S_s[tid * Bc + y] = __expf(S_s[tid * Bc + y] - row_m);
                row_l += S_s[tid * Bc + y];
            }
            float row_m_prev = m_start[i * Br + tid];
            float row_l_prev = l_start[i * Br + tid];

            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = __expf(row_m_prev - row_m_new) * row_l_prev + __expf(row_m - row_m_new) * row_l;

            for(int y = 0; y < d; y++){
                float pv = 0;
                for(int x = 0; x < Bc; x++){
                    pv += S_s[tid * Bc + x] * V_s[x * d + y];
                }
                O_start[i * Br * d + tid * d + y] = 1 / row_l_new * (row_l_prev * __expf(row_m_prev - row_m_new) * O_start[i * Br * d + tid * d + y] + __expf(row_m - row_m_new) * pv);
            }
            m_start[i * Br + tid] = row_m_new;
            l_start[i * Br + tid] = row_l_new;
        }
        __syncthreads();
    } 
}


torch::Tensor forward_learn(torch::Tensor Q, torch::Tensor K, torch::Tensor V){
    const int Bc = 32;
    const int Br = 32;

    const int B = Q.size(0);
    const int nh = Q.size(1);
    const int N = Q.size(2);
    const int d = Q.size(3);

    const int Tr = ceil((float)N / Br);
    const int Tc = ceil((float)N / Bc);
    const float softmax_scale = 1 / sqrt(d);

    auto O = torch::zeros_like(Q); // torch.zeros_like(input) is equivalent to torch.zeros(input.size(), dtype=input.dtype, layout=input.layout, device=input.device).
    auto m = torch::full({B, nh, N}, -INFINITY);
    auto l = torch::zeros({B, nh, N});
    torch::Device device(torch::kCUDA);
    m = m.to(device);
    l = l.to(device);
    
    const int sram_size = Br * d * sizeof(float) + 2 * Bc * d * sizeof(float) + Br * Bc * sizeof(float);
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("max sram size : %d, requested sram size: %d \n", max_sram_size, sram_size);

    dim3 grid_size(B, nh);
    dim3 block_size(Bc);

    forward_learn_kernel<<<grid_size, block_size, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Br, Bc, Tr, Tc, softmax_scale,
        m.data_ptr<float>(), l.data_ptr<float>(), O.data_ptr<float>()
    );
    cudaDeviceSynchronize();
    return O;
}