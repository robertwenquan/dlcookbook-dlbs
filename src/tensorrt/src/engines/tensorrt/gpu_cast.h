#ifndef DLBS_TENSORRT_BACKEND_ENGINES_GPU_CAST
#define DLBS_TENSORRT_BACKEND_ENGINES_GPU_CAST

void gpu_cast(int batch_size, int input_size, unsigned char* __restrict__ input, float* __restrict__ output, cudaStream_t stream);

#endif