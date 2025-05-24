# FlashAttention 小复现
这是一个根据b站up主“比飞鸟贵重的多_HKL”的FlahAttention视频来学习的[Flash Attention学习过程 - bilibili](https://www.bilibili.com/video/BV1FM9XYoEQ5/?spm_id_from=333.999.0.0&vd_source=1e1c4d48c6129699686897a835e568ea)

环境配置情况等更详细的情况可以到[我的小博客](https://wabbybabb0.github.io/2025/04/29/flashattention-mini/)小博客瞅瞅

## 关于调试
为了便于调试，使用libtorch——提供对PyTorch模型和算子的原生C++支持，可以在不依赖Python的情况下调试CUDA。(思路来自比飞鸟贵重的多_HKL)
## 关于代码
* `example-app.cpp`是测试libtorch是否可以用的例程，具体见[libtorch-demo测试](https://pytorch.org/cppdocs/installing.html)
* `falsh_learn.cu`是实现forward CUDA核心逻辑的文件
* `falsh-atten-main.cpp`是主程序
* `curand_mt19937_lognormal_example.cpp`是cuRAND测试代码
* `test-attention.cpp`是用libtorch实现Attention计算为FA1做检验用

## 关于检验

* 验证方法：使用libtorch做逻辑和精度验证，使用curand设置统一的随机数，[cuRAND mt19937 example - github](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuRAND/Host/mt19937)，打印结果第一个block第三个head的前256个数据对照(i=16384; i<16640)、第二个block第三个head的前256个数据对照(i=81920; i<82176)，检验通过

* 性能测试：也许是我的家用级显卡(3050)限制了FA1的发挥吧哈哈o(*￣▽￣*)ブ

  ```
  torch：0.638ms
  my FA1：36ms
  ```

  