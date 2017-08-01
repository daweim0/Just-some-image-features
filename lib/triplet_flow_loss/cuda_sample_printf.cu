

// System includes
#include <stdio.h>
#include <assert.h>

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

__global__ void testKernel(int val)
{
    printf("[%d, %d]:\t\tValue is:%d\n",\
            blockIdx.y*gridDim.x+blockIdx.x,\
            threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
            val);
}

int main(int argc, char **argv)
{
    int devID;
    cudaDeviceProp props;

    // This will pick the best possible CUDA capable device
//    devID = findCudaDevice(argc, (const char **)argv);

    //Get GPU information
//    checkCudaErrors(cudaGetDevice(&devID));
//    checkCudaErrors(cudaGetDeviceProperties(&props, devID));
    printf("Device %d: \"%s\" with Compute %d.%d capability\n",
           devID, props.name, props.major, props.minor);

    printf("printf() is called. Output:\n\n");

    //Kernel configuration, where a two-dimensional grid and
    //three-dimensional blocks are configured.
//    dim3 dimGrid(2, 2);
//    dim3 dimBlock(2, 2, 2);
    testKernel<<<3, 3>>>(10);
    cudaDeviceSynchronize();

    return 0;
}

