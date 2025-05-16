# FlashAttention 小复现
这是一个根据b站up主“比飞鸟贵重的多_HKL”的FlahAttention视频来学习的[Flash Attention学习过程 - bilibili](https://www.bilibili.com/video/BV1FM9XYoEQ5/?spm_id_from=333.999.0.0&vd_source=1e1c4d48c6129699686897a835e568ea)
## 关于调试
为了便于调试，使用libtorch——提供对PyTorch模型和算子的原生C++支持，可以在不依赖Python的情况下调试CUDA。(思路来自比飞鸟贵重的多_HKL)
## 关于代码
'example-app.cpp'是测试libtorch是否可以用的例程，具体见[libtorch-demo测试](https://pytorch.org/cppdocs/installing.html)
'falsh_learn.cu'是实现forward CUDA核心逻辑的文件
'falsh-atten-main.cpp'是主程序
