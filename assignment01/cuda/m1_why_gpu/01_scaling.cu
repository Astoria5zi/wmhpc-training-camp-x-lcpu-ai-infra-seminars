// 问题 1.5：并行度与吞吐。同一个向量加法，用四种方式跑一遍，
// 对比每种方式的耗时并解释差距。
// 编译运行：make run/m1_why_gpu/01_scaling
#include <chrono>
#include "common.h"

// —— Kernel 1 —— 整块 GPU 上只有一个线程在干活 ——
// 物理上 GPU 有数千个 core，但 launch <<<1,1>>> 只启动了 1 个线程。
// 这个线程要用 for 循环串行算完 400 万个加法，其余几千个 core 空转。
// 效果：比 CPU 还慢（因为 GPU 还有启动开销和内存传输延迟）
__global__ void add_one_thread(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
}

// —— Kernel 2 —— 一个 block、256 个线程 ——
// 启动了 256 个线程，它们同时执行这条 kernel。
// threadIdx.x = 本线程在 block 内的编号（0~255）
// blockDim.x   = block 的 x 维度大小（= 256）
// 每个线程负责处理 i, i+256, i+512, ... 这些元素（grid-stride 循环）
// 对比 <<<1,1>>>：256 倍的并行度，时间应该快大约 40 倍
__global__ void add_one_block(const float *a, const float *b, float *c, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) c[i] = a[i] + b[i];
}

// —— Kernel 3 —— 铺满整个 grid ——
// 启动了 blocks × 256 个线程（blocks ≈ n / 256）
// 通常每个线程只处理 1 个元素（一个线程管一个格子）
// blockIdx.x = block 编号（0 ~ blocks-1）
// blockDim.x = 每个 block 的线程数（256）
// threadIdx.x = 线程在该 block 内的编号
// i = blockIdx.x * blockDim.x + threadIdx.x 是每个线程的「身份证」
// 这是真正的「数据并行」—— 400 万个线程，每人算 1 个加法
// 结果：GPU 的真正实力
__global__ void add_grid(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    // —— 准备数据：4M 个 float（约 16 MB）——
    // 1 << 22 = 4194304，左移 22 位 = 2^22
    const int n = 1 << 22;  // 4M 元素
    size_t bytes = (size_t)n * sizeof(float);

    // h_ 开头 = host（CPU 端）内存
    float *h_a = (float *)malloc(bytes);   // 输入 A
    float *h_b = (float *)malloc(bytes);   // 输入 B
    float *h_c = (float *)malloc(bytes);   // 输出 C（GPU 运算结果）
    float *h_ref = (float *)malloc(bytes); // 输出 ref（CPU 算的标准答案，用来对拍）
    fill_random(h_a, n, 1);
    fill_random(h_b, n, 2);

    // —— CPU 单线程做向量加法（作为正确性基准）——
    // 用 C++ chrono 计时
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; i++) h_ref[i] = h_a[i] + h_b[i];
    auto t1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("CPU 单线程      : %10.3f ms  (%6.2f ns/元素)\n", cpu_ms,
           cpu_ms * 1e6 / n);

    // —— 在 GPU 显存上分配空间，把数据从 CPU 拷贝到 GPU ——
    // d_ 开头 = device（GPU 端）内存
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));      // GPU 上分配 16 MB
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));  // CPU → GPU
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // —— 热身（warmup）——
    // 第一次调用 kernel 有额外开销（加载指令、分配资源等），
    // 所以先跑一次不计时，让 GPU「热」起来
    int threads = 256;
    int blocks = (n + threads - 1) / threads;  // 向上取整，保证覆盖所有元素
    add_grid<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK_KERNEL();

    GpuTimer timer;  // common.h 里的计时工具

    // —— 测试 1：GPU <<<1, 1>>> ——
    // 启动 1 个 block × 1 个线程 = 总共 1 个线程在工作
    timer.start();
    add_one_thread<<<1, 1>>>(d_a, d_b, d_c, n);
    float ms1 = timer.stop_ms();
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));  // GPU → CPU
    if (!check_close(h_c, h_ref, n)) REPORT(0);  // 和 CPU 结果对拍，不对就退出
    printf("GPU <<<1, 1>>>  : %10.3f ms  (%6.2f ns/元素)\n", ms1, ms1 * 1e6 / n);

    // —— 测试 2：GPU <<<1, 256>>> ——
    // 启动 1 个 block × 256 个线程 = 总共 256 个线程在工作
    timer.start();
    add_one_block<<<1, 256>>>(d_a, d_b, d_c, n);
    float ms2 = timer.stop_ms();
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    if (!check_close(h_c, h_ref, n)) REPORT(0);
    printf("GPU <<<1, 256>>>: %10.3f ms  (%6.2f ns/元素)\n", ms2, ms2 * 1e6 / n);

    // —— 测试 3：GPU 铺满 grid ——
    // 启动 blocks 个 block × 256 线程 = 总共 n 个线程在工作
    // 这才是「满配」的 GPU，全部 SM 都跑满
    timer.start();
    add_grid<<<blocks, threads>>>(d_a, d_b, d_c, n);
    float ms3 = timer.stop_ms();
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    if (!check_close(h_c, h_ref, n)) REPORT(0);
    printf("GPU 铺满 grid   : %10.3f ms  (%6.2f ns/元素, %d blocks x %d threads)\n",
           ms3, ms3 * 1e6 / n, blocks, threads);

    REPORT(1);  // 打印 PASS
    return 0;
}
