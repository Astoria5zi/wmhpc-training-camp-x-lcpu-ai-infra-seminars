#include<cstdio>
#include<cstdlib>
#include<cuda_runtime.h>

#define CHECK(call) \
    do{\
        cudaError_t e_= (call); \
        if(e_ != cudaSuccess){ \
             printf("CUDA error at %s:%d: %s\n",__FILE__, __LINE__, cudaGetErrorString(e_));\
            return 1;\
        }\
    }while(0)\


__global__ void calculate(float *a, float *b, int n){
    
    int stride = gridDim.x * blockDim.x;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride ){
        b[i] = 2.0f * a[i] + b[i];
    }
}

int main(int argc, char *argv[]){

    const int n = atoi(argv[1]);
    if(n == 0){
        printf("SUM=0\n");
        return 0;
    }
    size_t bytes = (size_t)n * sizeof(float);

    
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);

    for(int i = 0; i < n; i ++){
        h_a[i] = ((i % 2048) - 1024) * 0.5f;
        h_b[i] = ((i % 1024) - 512);
    }


    float *d_a;
    float *d_b;
    // 检查
    CHECK(cudaMalloc(&d_a, bytes));
    CHECK(cudaMalloc(&d_b, bytes));
    

    // 检查
    CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    calculate<<<blocksPerGrid, threadsPerBlock>>> (d_a, d_b, n);
    CHECK(cudaGetLastError());        // 检查 kernel 启动参数错误（launch 配置）
    CHECK(cudaDeviceSynchronize());   // 等 kernel 跑完，同时检查执行错误

    // 拷回CPU + 检查
    CHECK(cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost));
    

    // 求和
    double sum = 0;
    for(int i = 0; i < n; i ++){
        sum += h_b[i];
    }

    printf("SUM=%.0f\n", sum);
    
    
    return 0;
}