// 问题 4.2：三点平均 stencil（填空）。
// out[i] = (in[i-1] + in[i] + in[i+1]) / 3，越界位置按 0 处理。
// 写两个 kernel：一个用静态 shared memory，一个用动态的。
// 填完之前这个文件无法通过编译。
// 注：设置block内共享tile数组是为了减少对global memory的访问次数——
//    每个元素只从显存读一次，块内的三次复用都走片上的 shared memory
//    （延迟远低于显存）。
#include "common.h"

#define BLOCK 256
#define RADIUS 1

// ============================================================
// 静态 shared 版本：tile 大小在编译期写死在声明里
// 思想：每个 block 把「自己的 BLOCK 个元素 + 左右各 RADIUS 个邻居」
//       一次性搬进 shared，之后三点平均全部走片上，显存只读一次
// ============================================================
__global__ void stencil_static(const float *in, float *out, int n) {
    // // 静态 shared 数组：BLOCK 个自己的 + 左 RADIUS 个 + 右 RADIUS 个
    // // RADIUS=1 → 256 + 2 = 258 个 float ≈ 1KB，编译期固定
    __shared__ float tile[BLOCK + 2 * RADIUS];

    // g = 全局下标：第几个 block（blockIdx.x）× 每 block 线程数 + 线程内编号
    //   例：block 3, tid 5 → g = 3*256 + 5 = 773
    int g = blockIdx.x * blockDim.x + threadIdx.x;

    // l = tile 内的局部下标：tid + RADIUS
    //   为什么 +RADIUS？给左 halo 让位——tid 0 不占 tile[0]，占 tile[1]
    //   好处：所有线程统一偏移，边界线程不用特判
    int l = threadIdx.x + RADIUS;

    // 主搬运：每个线程把自己的元素从显存搬进 shared
    //   (g < n) 处理数组尾部的越界：最后不足 BLOCK 个时，多出的线程写 0
    tile[l] = (g < n) ? in[g] : 0.f;

    // halo 搬运：只有块两端的线程（tid < RADIUS，即 tid 0）额外多搬 2 个邻居
    if(threadIdx.x < RADIUS){
        int left = g - RADIUS;
        int right = g + BLOCK;
        tile[l - RADIUS] = (left >= 0) ? in[left] : 0.f;
        tile[l + RADIUS] = (right < n) ? in[right] : 0.f;
    }

    // 同步屏障：必须等【所有】线程搬完（含 halo），才能开始读邻居
    // 不然后面的线程可能读到别人还没写入的旧值（module3 的教训）
    __syncthreads();

    if (g < n) {
        // 三点平均：全部从 tile 读（不许碰 in）
        //   l-1 = 左邻居，l = 自己，l+1 = 右邻居
        //   边界线程也安全：l=1 时 l-1=0（左 halo），l=256 时 l+1=257（右 halo）
        out[g] = (tile[l - 1] + tile[l] + tile[l + 1]) / 3.f;
    }
}

// ============================================================
// 动态 shared 版本：逻辑和静态版完全一样，唯一区别是
//   tile 大小不在声明里，而在 launch 时用第三个参数指定
// ============================================================
__global__ void stencil_dynamic(const float *in, float *out, int n) {
    // 动态 shared 声明：extern + 空方括号，编译期不确定大小
    //   实际大小 = launch 时的第三个参数（每 block 分多少字节）
    extern __shared__ float tile[];

    int g = blockIdx.x * blockDim.x + threadIdx.x;   // 同静态版
    int l = threadIdx.x + RADIUS;                    // 同静态版

    tile[l] = (g < n) ? in[g] : 0.f;                 // 主搬运，同静态版
    if (threadIdx.x < RADIUS) {                      // halo 搬运，同静态版
        int left = g - RADIUS;
        int right = g + BLOCK;
        tile[l - RADIUS] = (left >= 0) ? in[left] : 0.f;
        tile[l + BLOCK] = (right < n) ? in[right] : 0.f;
    }
    __syncthreads();                                 // 同步，同静态版
    if (g < n) {
        out[g] = (tile[l - 1] + tile[l] + tile[l + 1]) / 3.f;  // 同静态版
    }
    // launch 时的第三参数：(BLOCK + 2*RADIUS) * sizeof(float)
    //   = 258 * 4 = 1032 字节 —— 这是【每个 block】分到的动态 shared 大小
}

int main() {
    const int n = 1000003;
    size_t bytes = (size_t)n * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    fill_random(h_in, n, 3);
    for (int i = 0; i < n; i++) {
        float l = (i > 0) ? h_in[i - 1] : 0.f;
        float r = (i < n - 1) ? h_in[i + 1] : 0.f;
        h_ref[i] = (l + h_in[i] + r) / 3.f;
    }

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int blocks = (n + BLOCK - 1) / BLOCK;

    stencil_static<<<blocks, BLOCK>>>(d_in, d_out, n);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    if (!check_close(h_out, h_ref, n)) REPORT(0);
    printf("static  PASS\n");

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    // ====== 空 5：动态 shared 版本的 launch——第三个参数该填多少字节？ ======
    stencil_dynamic<<<blocks, BLOCK, (BLOCK + 2 * RADIUS) * sizeof(float)/* 填这里 */>>>(d_in, d_out, n);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    if (!check_close(h_out, h_ref, n)) REPORT(0);
    printf("dynamic PASS\n");

    REPORT(1);
    return 0;
}
