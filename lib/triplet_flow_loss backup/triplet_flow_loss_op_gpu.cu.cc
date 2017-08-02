#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <cfloat>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <thrust/device_vector.h>
//#include <curand.h>
//#include <curand_kernel.h>
#include "triplet_flow_loss_op_gpu.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

// namespace tensorflow {
using namespace tensorflow;

inline
cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}


int intRand_cu(const int & min, const int & max) {
    static thread_local std::mt19937 generator;
    std::uniform_int_distribution<int> distribution(min,max);
    return distribution(generator);
}


int crop_val_cu(int a, int min, int max) {
    if(a < min)
        return min;
    else if(a > max)
        return max;
    else
        return a;
}


//__global__ void init_state(const int nthreads, unsigned int seed, curandState_t* states)
//{
//  CUDA_1D_KERNEL_LOOP(index, nthreads)
//  {
//    curand_init(seed, index, 0, &states[index]);
//  }
//}


template <typename Dtype>
__global__ void TripletForward(const int output_size, const Dtype* data_l, const Dtype* data_r, int* triplets,
    const int n_triplets, const int feature_depth, const float margin, Dtype* losses, Dtype* diff_l, Dtype* diff_r)
{
    CUDA_1D_KERNEL_LOOP(index, n_triplets)
    {
        printf("starting kernel loop at index %i out of %i n_triplets\n", index, n_triplets);
        // compute the distances
        int index_i = triplets[index * 3 + 0];
        int index_j = triplets[index * 3 + 1];
        int index_k = triplets[index * 3 + 2];

        Dtype D_ij = 0;
        Dtype D_ik = 0;
        for (int c = 0; c < feature_depth; c++)
        {
            int data_i_index = index_i * feature_depth + c;
            int data_j_index = index_j * feature_depth + c;
            int data_k_index = index_k * feature_depth + c;
            if(!(0 <= data_i_index && data_i_index < output_size * feature_depth)) {printf("data_i_index: %i, output_size * feature_depth: %i, index=%i, n_triplets=%i\n", data_i_index, output_size * feature_depth, index, n_triplets);}
            if(!(0 <= data_j_index && data_j_index < output_size * feature_depth)) {printf("data_j_index: %i, output_size * feature_depth: %i, index=%i, n_triplets=%i\n", data_j_index, output_size * feature_depth, index, n_triplets);}
            if(!(0 <= data_k_index && data_k_index < output_size * feature_depth)) {printf("data_k_index: %i, output_size * feature_depth: %i, index=%i, n_triplets=%i\n", data_k_index, output_size * feature_depth, index, n_triplets);}

            D_ij += powf(data_l[data_i_index] - data_r[data_j_index], 2);
            D_ik += powf(data_l[data_i_index] - data_r[data_k_index], 2);
        }

        // store the loss
        Dtype dis = D_ij - D_ik + margin;
        losses[index] = max(dis, Dtype(0.0));

        // compute gradients
        if (dis > 0)
        {
            for (int c = 0; c < feature_depth; c++)
            {
                int data_i_index = index_i * feature_depth + c;
                int data_j_index = index_j * feature_depth + c;
                int data_k_index = index_k * feature_depth + c;
                if(!(0 <= data_i_index && data_i_index < output_size * feature_depth)) {printf("data_i_index: %i, output_size * feature_depth: %i, index=%i, n_triplets=%i\n", data_i_index, output_size * feature_depth, index, n_triplets);}
                if(!(0 <= data_j_index && data_j_index < output_size * feature_depth)) {printf("data_j_index: %i, output_size * feature_depth: %i, index=%i, n_triplets=%i\n", data_j_index, output_size * feature_depth, index, n_triplets);}
                if(!(0 <= data_k_index && data_k_index < output_size * feature_depth)) {printf("data_k_index: %i, output_size * feature_depth: %i, index=%i, n_triplets=%i\n", data_k_index, output_size * feature_depth, index, n_triplets);}

                // update x_i
                diff_l[data_i_index] +=
                                (data_l[data_k_index] - data_r[data_j_index]) / output_size;
                // update x_j
                diff_r[data_j_index] +=
                                (data_r[data_j_index] - data_l[data_i_index]) / output_size;
                // update x_k
                diff_l[data_k_index] +=
                              (data_l[data_i_index] - data_l[data_k_index]) / output_size;
            }
        }
    }
}

template <typename Dtype>
__global__ void sum_gradients(const int channels, const Dtype* diffs_l, const Dtype* diffs_r, const int* triplets,
    const int batch_size, const int height, const int width, Dtype* bottom_l, Dtype* bottom_r)
{
  CUDA_1D_KERNEL_LOOP(index, batch_size * height * width * channels)
  {
      bottom_l[index] = diffs_l[index];
      bottom_r[index] = diffs_r[index];
  }
}

// bottom_data: (batch_size, height, width, channels)
bool TripletForwardLaucher(
    const float* left_data, const float* right_data, const float* flow_tensor, const int* occluded_tensor,
	const int batch_size, const int height, const int width, const int channels, const int n_triplets,
	const float margin, const int negative_radius_, float* top_data, float* bottom_left_diff, float* bottom_right_diff, const Eigen::GpuDevice& d) {

    cudaError_t err;
    int saftey_margin = 3;

    // copy flow to CPU
    int flow_arr_size = batch_size * height * width * 2;
    float* flow_array = (float*)malloc(flow_arr_size * sizeof(float));
    cudaMemcpy(flow_array, flow_tensor, batch_size * height * width * sizeof(int), cudaMemcpyDeviceToHost);

    // copy occlusiuons to CPU
    int* occluded_array = (int*)malloc(batch_size * height * width * sizeof(int));
    cudaMemcpy(occluded_array, occluded_tensor, batch_size * height * width * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "test0" << std::endl;

    const int output_size = batch_size * height * width;

    // sampling
    std::srand ( unsigned ( std::time(0) ) );
    std::vector<int> triplets(batch_size * n_triplets * 3);
    for (int n = 0; n < batch_size; n++)
    {
        for(int triplet_num = 0; triplet_num < n_triplets; triplet_num++)
        {
            int h = std::rand() % height;
            int w = std::rand() % width;

            // anchor
            int index = (n * height + h) * width + w;
            if(occluded_array[index] != 0)
            {
                continue;  // don't use this point
            }

            // find the corresponding pixel
            if(index + 1 >= flow_arr_size) {
                std::cout << "index greater than flow array" << std::endl;
                std::cout << "\tflow_arr_size: " << flow_arr_size << std::endl;
                std::cout << "\tindex*2+1: " << index*2+1 << std::endl;
                assert(false);
            }
            float w_positive = w + round(flow_array[index * 2]);
            float h_positive = h + round(flow_array[index * 2 + 1]);
            if(0 > w_positive || 0 > h_positive || width <= w_positive || height <= h_positive) {
    //                        std::cout << "flow goes outside image, h: " << h << " w: " << w << std::endl;
                continue;  // corresponding right image location is outside image
            }
            int index_positive = n * height * width + h_positive * width + w_positive;
            if(!(-1 < index_positive && index_positive < output_size)) {
                continue;
            }

            // sample a negative pixel
            // check the predicted label of this pixel for hard negative

            int h_displacement = intRand_cu(0, negative_radius_*2 - saftey_margin) - negative_radius_;
            if(-1 * saftey_margin < h_displacement){
                h_displacement += saftey_margin * 2;
            }
            int h_negative = crop_val_cu(h_displacement + h_positive, 0, height - 1);

            int w_displacement = intRand_cu(0, negative_radius_*2 - saftey_margin) - negative_radius_;
            if(-1 * saftey_margin < w_displacement){
                w_displacement += saftey_margin * 2;
            }
            int w_negative = crop_val_cu(w_displacement + w_positive, 0, height - 1);

            int index_negative = ((n) * height + h_negative) * width + w_negative;

            if(!(-1 < index && index < batch_size * height * width)) {
                printf("assertion on line __LINE__, index: %i", index);
            }
            if(!(-1 < index_positive && index_positive < batch_size * height * width)) {
                printf("assertion on line __LINE__, index_positive: %i", index_positive);
            }
            if(!(-1 < index_negative && index_negative < batch_size * height * width)) {
                printf("assertion on line __LINE__, index_negative: %i", index_negative);
            }

            // store the triplet
            triplets.push_back(index);
            triplets.push_back(index_positive);
            triplets.push_back(index_negative);
        }
    }

    std::cout << "test1" << std::endl;
    // run kernels
    const int kThreadsPerBlock = 10;
    std::cout << "output_size: " << output_size << " batch_size: " << batch_size << " height: " << height << " width: " << width << std::endl;

    int actual_n_triplets = triplets.size() / 3;

    std::vector<int> test_vect;
    for(int i = 0; i < triplets.size(); i++) {
        test_vect.push_back(i);
    }

    // compute the loss matrix
    float* losses;
    float* diffs_l;
    float* diffs_r;
    int* triplets_device;
    checkCuda(cudaMalloc((void **) &losses, actual_n_triplets * sizeof(float)));
//    checkCuda(cudaMalloc((void **) &diffs_l, output_size * channels * 3 * sizeof(float)));
//    checkCuda(cudaMalloc((void **) &diffs_r, output_size * channels * 3 * sizeof(float)));
    cudaMemset(bottom_left_diff, 0, batch_size * height * width * channels * sizeof(float));
    cudaMemset(bottom_right_diff, 0, batch_size * height * width * channels * sizeof(float));
    checkCuda(cudaMalloc((void **) &triplets_device, triplets.size() * sizeof(int)));
    cudaMemcpy(triplets_device, test_vect.data(), triplets.size() * sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemset(diffs_l, 0, output_size * channels * 3 * sizeof(float));
//    cudaMemset(diffs_r, 0, output_size * channels * 3 * sizeof(float));

    std::cout << "test2" << std::endl;

    TripletForward<<<40, 40>>>(
      output_size, left_data, right_data, triplets_device, actual_n_triplets, channels, margin, losses, bottom_left_diff, bottom_right_diff);

    std::cout << "test3" << std::endl;
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if(cudaSuccess != err) {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    // sum the loss and diffs
    thrust::device_ptr<float> losses_ptr(losses);
    float loss = thrust::reduce(losses_ptr, losses_ptr + actual_n_triplets);
    loss /= actual_n_triplets * 2.0;
    cudaMemcpy(top_data, &loss, sizeof(float), cudaMemcpyHostToDevice);

//    cudaMemset(bottom_left_diff, 0, batch_size * height * width * channels * sizeof(float));
//    cudaMemset(bottom_right_diff, 0, batch_size * height * width * channels * sizeof(float));
    std::cout << "test4" << std::endl;
//    sum_gradients<<<40, 40>>>(
//      channels, diffs_l, diffs_r, triplets_device, batch_size, height, width, bottom_left_diff, bottom_right_diff);
    std::cout << "test5" << std::endl;
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if(cudaSuccess != err) {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    // clean up
    free(flow_array);
    free(occluded_array);
    cudaFree(losses);
//    cudaFree(diffs_l);
//    cudaFree(diffs_r);
    cudaFree(triplets_device);
    std::cout << "test6" << std::endl;

    return d.ok();
}


template <typename Dtype>
__global__ void TripletBackward(const int nthreads, const Dtype* top_diff,
    const Dtype* bottom_diff_l, const Dtype* bottom_diff_r, Dtype* output_l, Dtype* output_r)
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    output_l[index] = top_diff[0] * bottom_diff_l[index];
    output_r[index] = top_diff[0] * bottom_diff_r[index];
  }
}

 
bool TripletFlowBackwardLaucher(const float* top_diff, const float* bottom_left_diff, const float* bottom_right_diff, const int batch_size,
	const int height, const int width, const int channels, float* output_l, float* output_r, const Eigen::GpuDevice& d)
{
  const int kThreadsPerBlock = 1024;
  const int output_size = batch_size * height * width * channels;
  cudaError_t err;

    std::cout << "about to call gradient kernel" << std::endl;
  TripletBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, top_diff, bottom_left_diff, bottom_right_diff, output_l, output_r);
    std::cout << "finished calling gradient kernel" << std::endl;
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return d.ok();
}

// }  // namespace tensorflow

#endif  // GOOGLE_CUDA
