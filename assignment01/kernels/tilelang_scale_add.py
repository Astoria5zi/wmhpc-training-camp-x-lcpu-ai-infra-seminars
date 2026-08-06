"""问题 7.3：TileLang 版 scale-add（填空）。

Y = 2 * X + 1，X 形状 (M, N)。两个空对应 TileLang 的两个 basic operation。
需要 GPU 和 tilelang（uv sync --extra tilelang），在集群上运行：
    pytest tests/test_tilelang.py -k scale_add
"""

import tilelang
import tilelang.language as T


def make_scale_add(M, N, block_M=32, block_N=32, dtype="float32"):
    @T.prim_func
    def scale_add(
        X: T.Buffer((M, N), dtype),
        Y: T.Buffer((M, N), dtype),
    ):
        # ====== 空 1：二维 CTA grid——x 方向要多少个 block（管 N 列），
        #         y 方向要多少个（管 M 行）？提示：T.ceildiv ======
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            # ====== 空 2：block 内并行遍历 tile 的每个元素，
            #         提示：T.Parallel(维度1, 维度2) ======
            for i, j in T.Parallel(block_M, block_N):
                gi = by * block_M + i
                gj = bx * block_N + j
                if gi < M and gj < N:
                    Y[gi, gj] = X[gi, gj] * 2.0 + 1.0

    return scale_add

# def make_scale_add(M, N, block_M=32, block_N=32, dtype="float32"):
#     # 工厂函数：根据形状 M×N 生成一个 TileLang kernel（类似 CUDA 里
#     # 根据 n 动态算 grid/block 尺寸，只是这里返回一个可编译的函数对象）

#     @T.prim_func                       # 标记这是 TileLang 的 primitive function
#     def scale_add(                     # （相当于 __global__，但声明的是"数据流"而非线程）
#         X: T.Buffer((M, N), dtype),    # 声明全局输入 buffer：形状 M×N，float32
#         Y: T.Buffer((M, N), dtype),    # 声明全局输出 buffer（同形状）
#     ):
#         # ---- 空1：启动配置（对应 CUDA 的 <<<grid, block>>>）----
#         with T.Kernel(T.ceildiv(N, block_N),   # grid 的 x 维 = N 列按 block_N=32 切成几段
#                       T.ceildiv(M, block_M),   # grid 的 y 维 = M 行按 block_M=32 切成几段
#                       threads=128) as (bx, by):  # 每 block 128 线程；bx=列向 block 号，by=行向 block 号
#             # 进入 with 块 = 一个 CTA（≈CUDA 的一个 block）开始执行

#             # ---- 空2：block 内并行遍历 tile 的每个元素 ----
#             for i, j in T.Parallel(block_M, block_N):
#                 # T.Parallel(32, 32) = 一个 32×32=1024 次的并行循环，
#                 # 编译器自动摊到 128 个线程上（每线程 8 次），
#                 # 等价于 CUDA 里你手写 threadIdx 展开循环
#                 gi = by * block_M + i   # 行号：当前 block 起始行(by*32) + block 内行偏移 i
#                 gj = bx * block_N + j   # 列号：当前 block 起始列(bx*32) + block 内列偏移 j
#                 if gi < M and gj < N:   # 边界守卫：末尾 block 的部分元素会越界（如 M=123 时
#                     Y[gi, gj] = X[gi, gj] * 2.0 + 1.0   # gi=127 越界），必须拦
#                 # 计算只有一行：跟 CUDA 里 Y[i] = X[i]*2+1 完全同构

#     return scale_add                   # 返回函数对象，交给 tilelang.compile(func, out_idx=[1])

