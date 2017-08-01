/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Lifted Structured Loss Op


//#define DEBUG_NAN

const int n_target_triplets = 40;  // per object
const bool dense_sampling = false;
const int MAX_ITERS = 10000;


#include <stdio.h>
#include <cfloat>
#include <math.h>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <unistd.h>
#include <fenv.h>

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;


int crop_val(int a, int min, int max) {
    if(a < min)
        return min;
    else if(a > max)
        return max;
    else
        return a;
}

void do_nothing(int foo){}


uint32_t state[4] = {1234567890, 1123456789, 777777777, 'g'*'r'*'8'+'m'*8};

// from https://en.wikipedia.org/wiki/Xorshift
/* The state array must be initialized to not be all zero */
uint32_t xorshift128()
{
    uint32_t t = state[3];
    t ^= t << 11;
    t ^= t >> 8;
    state[3] = state[2]; state[2] = state[1]; state[1] = state[0];
    t ^= state[0];
    t ^= state[0] >> 19;
    state[0] = t;
    return t;
}


REGISTER_OP("TripletFlow")
.Attr("T: {float, double}")
.Attr("margin: float")
.Attr("negative_radius: int")
.Input("left_data: T")
.Input("right_data: T")
.Input("gt_flow: T")
.Input("occluded_mask: int32")
.Input("left_mask: int32")
.Input("right_mask: int32")
.Output("loss: T")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
c->set_output(0, c->Scalar());
return Status::OK();
})
.Output("bottom_left_diff: T")
.Output("bottom_right_diff: T");

REGISTER_OP("TripletFlowGrad")
.Attr("T: {float, double}")
.Attr("margin: float")
.Input("bottom_left_diff: T")
.Input("bottom_right_diff: T")
.Input("grad: T")
.Output("output_left: T")
.Output("output_right: T");

template <typename Device, typename T>
class TripletFlowOp : public OpKernel {
public:
    explicit TripletFlowOp(OpKernelConstruction* context) : OpKernel(context) {
        // Get the margin
        OP_REQUIRES_OK(context,
                       context->GetAttr("margin", &margin_));
        // Check that margin is positive
        OP_REQUIRES(context, margin_ >= 0,
                    errors::InvalidArgument("Need margin >= 0, got ", margin_));
        // Get the margin
        OP_REQUIRES_OK(context,
                       context->GetAttr("negative_radius", &negative_radius_));
        // Check that margin is positive
        OP_REQUIRES(context, negative_radius_ >= 0,
                    errors::InvalidArgument("Need negative radius >= 0, got ", negative_radius_));
    }

    // bottom_data: (batch_size, height, width, channels)
    // bottom_label: (batch_size, height, width, num_classes)
    void Compute(OpKernelContext* context) override
    {
        signal(SIGUSR1, do_nothing);

        // Grab the input tensor
        const Tensor& left_data = context->input(0);
        const T* left_data_array = left_data.flat<T>().data();

        const Tensor& right_data = context->input(1);
        const T* right_data_array = right_data.flat<T>().data();

        const Tensor& flow_tensor = context->input(2);
        const T* flow_array = flow_tensor.flat<T>().data();

        const Tensor& occluded = context->input(3);
        const int* occluded_array = occluded.flat<int>().data();

        const Tensor& left_mask = context->input(4);
        const int* left_mask_array = left_mask.flat<int>().data();

        const Tensor& right_mask = context->input(5);
        const int* right_mask_array = right_mask.flat<int>().data();

        // data should have 4 dimensions.
        OP_REQUIRES(context, left_data.dims() == 4,
                    errors::InvalidArgument("left data must be 3-dimensional"));

        OP_REQUIRES(context, right_data.dims() == 4,
                    errors::InvalidArgument("right data must be 3-dimensional"));

        OP_REQUIRES(context, flow_tensor.dims() == 4,
                    errors::InvalidArgument("flow must be 3-dimensional"));

        OP_REQUIRES(context, occluded.dims() == 4,
                    errors::InvalidArgument("occlusions must be 3-dimensional"));

        OP_REQUIRES(context, left_mask.dims() == 4,
                    errors::InvalidArgument("left mask must be 3-dimensional"));

        OP_REQUIRES(context, right_mask.dims() == 4,
                    errors::InvalidArgument("right mask must be 3-dimensional"));

        // batch size
        int batch_size = left_data.dim_size(0);
        // height
        int height = left_data.dim_size(1);
        // width
        int width = left_data.dim_size(2);
        // number of channels
        int feature_depth = left_data.dim_size(3);
        // number of object masks
        int n_object_masks = left_mask.dim_size(3);

//        std::cout << "Image batch_size: " << batch_size << " height: " << height << " width: " << width << " feature_depth: " << feature_depth << std::endl;
//        std::cout << "flow batch_size: " << flow_tensor.dim_size(0) << " height: " << flow_tensor.dim_size(1) << " width: " << flow_tensor.dim_size(2) << " feature_depth: " << flow_tensor.dim_size(3) << std::endl;
//        std::cout << "occluded batch_size: " << occluded.dim_size(0) << " height: " << occluded.dim_size(1) << " width: " << occluded.dim_size(2) << std::endl;

        assert(batch_size == flow_tensor.dim_size(0));
        assert(batch_size == occluded.dim_size(0));
        assert(batch_size == left_mask.dim_size(0));
        assert(batch_size == right_mask.dim_size(0));
        assert(height == flow_tensor.dim_size(1));
        assert(height == occluded.dim_size(1));
        assert(height == left_mask.dim_size(1));
        assert(height == right_mask.dim_size(1));
        assert(width == flow_tensor.dim_size(2));
        assert(width == occluded.dim_size(2));
        assert(width == left_mask.dim_size(2));
        assert(width == right_mask.dim_size(2));

#ifdef DEBUG_NAN
        feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);
#endif

        int feature_arr_size = batch_size * height * width * feature_depth;

        int flow_arr_size = flow_tensor.dim_size(0) * flow_tensor.dim_size(1) * flow_tensor.dim_size(2) * flow_tensor.dim_size(3);

        // Create output loss tensor
        int dim = 1;
        TensorShape output_shape;
//		TensorShapeUtils::MakeShape(&dim, 1, &output_shape);

        Tensor* top_data_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &top_data_tensor));
        auto top_data = top_data_tensor->template flat<T>();

        // bottom diff
        TensorShape output_shape_left_diff = left_data.shape();
        Tensor* bottom_left_diff_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, output_shape_left_diff, &bottom_left_diff_tensor));
        T* bottom_left_diff = bottom_left_diff_tensor->template flat<T>().data();
        memset(bottom_left_diff, 0, batch_size * height * width * feature_depth *sizeof(T));

        TensorShape output_shape_right_diff = left_data.shape();
        Tensor* bottom_right_diff_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(2, output_shape_right_diff, &bottom_right_diff_tensor));
        T* bottom_right_diff = bottom_right_diff_tensor->template flat<T>().data();
        memset(bottom_right_diff, 0, batch_size * height * width * feature_depth *sizeof(T));

        // sample triplets to define the loss
        // compute label indexes

//		// classes in the batch

        int saftey_margin = 2;  // the radius around the target where we shouldn't sample
        // sampling
        std::vector<int> triplets;  // don't initialize it with a size because some points might be occluded

        if(dense_sampling) {
            triplets.reserve(batch_size * height * width * 3);
            for (int n = 0; n < batch_size; n++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        // anchor
                        int index = n * height * width + h * width + w;
                        if (occluded_array[index] != 0) {
//                        std::cout << "point occluded, h: " << h << " w: " << w << std::endl;
                            continue;  // don't use this point
                        }

                        // find the corresponding pixel
                        if (index * 2 + 1 >= flow_arr_size) {
                            std::cout << "index greater than flow array" << std::endl;
                            std::cout << "\tflow_arr_size: " << flow_arr_size << std::endl;
                            std::cout << "\tindex*2+1: " << index * 2 + 1 << std::endl;
                            assert(false);
                        }
                        float w_positive = w + round(flow_array[index * 2]);
                        float h_positive = h + round(flow_array[index * 2 + 1]);
                        if (0 > w_positive || 0 > h_positive || width <= w_positive || height <= h_positive) {
//                        std::cout << "flow goes outside image, h: " << h << " w: " << w << std::endl;
                            continue;  // corresponding right image location is outside image
                        }
                        int index_positive = n * height * width + h_positive * width + w_positive;

                        // sample a negative pixel
                        // check the predicted label of this pixel for hard negative


                        int h_displacement = xorshift128() % (negative_radius_ * 2 - saftey_margin) - negative_radius_;
                        if (-1 * saftey_margin < h_displacement) {
                            h_displacement += saftey_margin * 2;
                        }
                        int h_negative = crop_val(h_displacement + h_positive, 0, height - 1);

                        int w_displacement = xorshift128() % (negative_radius_ * 2 - saftey_margin) - negative_radius_;
                        if (-1 * saftey_margin < w_displacement) {
                            w_displacement += saftey_margin * 2;
                        }
                        int w_negative = crop_val(w_displacement + w_positive, 0, width - 1);

                        int index_negative = ((n) * height + h_negative) * width + w_negative;

                        // store the triplet
                        triplets.push_back(index);
                        triplets.push_back(index_positive);
                        triplets.push_back(index_negative);
                        assert(-1 < index && index < batch_size * height * width);
                        assert(-1 < index_positive && index_positive < batch_size * height * width);
                        assert(-1 < index_negative && index_negative < batch_size * height * width);
                    }
                }
            }
        }

        else {
//            raise(SIGUSR1);

            triplets.reserve(batch_size * n_target_triplets * 5 * 3); // there are normaly <= 5 objects in an image
            for (int n = 0; n < batch_size; n++) {

                int n_objects_present = 0;
                std::vector<int> object_present;
                for(int i = 0; i < n_object_masks; i++) {
                    object_present.push_back(0);
                }
                for(int j = (n) * height * width; j < (n + 1) * height * width; j++) {
                    for(int i = 0; i < n_object_masks; i++) {
//                    raise(SIGUSR1);
                        if(left_mask_array[j * n_object_masks + i] != 0){
                            if(object_present[i] == 0) {
                                n_objects_present += 1;
                                object_present[i] = 1;
                            }
                        }
                    }
                }
                assert(n_objects_present != 0);
//                raise(SIGUSR1);

                for(int current_mask_index = 1; current_mask_index < n_object_masks; current_mask_index++) {
                    if(object_present[current_mask_index] == 0) {
                        assert(left_mask_array[(n) * height * width * n_object_masks + current_mask_index] == 0);
                    }
                    else {
                        int n_starting_triplets = triplets.size();
                        for (int iter = 0; (triplets.size() - n_starting_triplets < n_target_triplets) && (iter < n_target_triplets * 2); iter++) {
                            int h = 0;
                            int w = 0;

                            // anchor
                            int index = -1;
                            int index_positive = -1;
                            for (int i = 0; i < MAX_ITERS; i++) {
                                h = xorshift128() % height;
                                w = xorshift128() % width;
                                int index_ = ((n) * height + h) * prange(shape_0, nogil=True, schedule='static', num_threads=10):
            for b in range(shape_1):
                if occluded_m[a, b] == 1:
                    flow_depth_m[a, b, 0] = 0.0
                    flow_depth_m[a, b, 1] = 0.0
                    continue

                index_1d = a*shape_1+b
                end_x = <int> x_m[0, index_1d]
                end_y = <int> y_m[0, index_1d]

                if not ((0 <= end_x < shape_1) and (0 <= end_y < shape_0)):
                    flow_depth_m[a, b, 0] = x_m[0, index_1d] - b
                    flow_depth_m[a, b, 1] = y_m[0, index_1d] - a
                elif (end_point_depths_m[end_y, end_x, 0] < z_m[0, index_1d]) and (end_point_depths_m[end_y, end_x, 0] != 0):
                    occluded_m[a, b] = 1
                elif (end_point_depths_m[end_y, end_x, 0] > z_m[0, index_1d]) and (end_point_depths_m[end_y, end_x, 0] != 0):
                    occluded_m[<int> end_point_depths_m[end_y, end_x, 1], <int> end_point_depths_m[end_y, end_x, 2]] = 1
                    flow_depth_m[a, b, 0] = x_m[0, index_1d] - b
                    flow_depth_m[a, b, 1] = y_m[0, index_1d] - a
                    end_point_depths_m[end_y, end_x, 0] = z_m[0, index_1d]
                    im_left_warped_m[end_y, end_x, 0] = im_left_processed_m[a, b, 0]
                    im_left_warped_m[end_y, end_x, 1] = im_left_processed_m[a, b, 1]
                    im_left_warped_m[end_y, end_x, 2] = im_left_processed_m[a, b, 2]
                else:
                    flow_depth_m[a, b, 0] = x_m[0, index_1d] - b
                    flow_depth_m[a, b, 1] = y_m[0, index_1d] - a
                    end_point_depths_m[end_y, end_x, 0] = z_m[0, index_1d]
                    im_left_warped_m[end_y, end_x, 0] = im_left_processed_m[a, b, 0]
                    im_left_warped_m[end_y, end_x, 1] = im_left_processed_m[a, b, 1]
                    im_left_warped_m[end_y, end_x, 2] = im_left_processed_m[a, b, width + w;
                                if ((left_mask_array[(index_) * n_object_masks + current_mask_index] == 0) ||
                                                                                    (occluded_array[index_] != 0)) {
                                    continue;
                                }

                                // find the positive point
                                if (index_ * 2 + 1 >= flow_arr_size) {
                                    std::cout << "index greater than flow array" << std::endl;
                                    std::cout << "\tflow_arr_size: " << flow_arr_size << std::endl;
                                    std::cout << "\tindex*2+1: " << index_ * 2 + 1 << std::endl;
                                    assert(false);
                                }
                                int w_positive = w + round(flow_array[index_ * 2]);
                                int h_positive = h + round(flow_array[index_ * 2 + 1]);
                                if (0 > w_positive || 0 > h_positive || width <= w_positive || height <= h_positive) {
//                        std::cout << "flow goes outside image, h: " << h << " w: " << w << std::endl;
                                    continue;  // corresponding right image location is outside image
                                }
                                int index_positive_ = ((n) * height + h_positive) * width + w_positive;

                                if (right_mask_array[(index_positive_) * n_object_masks + current_mask_index] == 0) {
                                    continue;
                                }
                                assert((n * height * width <= index_positive_) && (index_positive_ < (n+1) * height * width));

                                index = index_;
                                index_positive = index_positive_;

                                break;  // the correct object isn't being sampled
                            }
                            if (index == -1) {
//                                std::cout << "(ind " << current_mask_index << ", iter " << iter << ")";
                                continue;
                            }
                            assert((n * height * width <= index) && (index < (n+1) * height * width));


                            // find a negative point on the correct object
                            int index_negative = -1;
                            for (int iter_neg = 0; iter_neg < MAX_ITERS; iter_neg++) {
                                // sample a negative pixel
                                int h_negative = xorshift128() % height;
                                int w_negative = xorshift128() % width;
                                int index_negative_ = ((n) * height + h_negative) * width + w_negative;

                                if (left_mask_array[(index_negative_) * n_object_masks + current_mask_index] != 0) {
                                    index_negative = index_negative_;
                                    break;  // found a point from the correct object
                                }
                            }
                            if (index_negative == -1) {
//                                std::cout
//                                        << "\tran into iteration limit looking for a negative point within object mask ("
//                                        << current_mask_index << ")" << std::endl;
                                continue;  // didn't find the a point from the correct object
                            }
                            assert((n * height * width <= index_negative) && (index_negative < (n+1) * height * width));


                            // store the triplet
                            triplets.push_back(index);
                            triplets.push_back(index_positive);
                            triplets.push_back(index_negative);

                            assert(-1 < index && index < batch_size * height * width);
                            assert(-1 < index_positive && index_positive < batch_size * height * width);
                            assert(-1 < index_negative && index_negative < batch_size * height * width);
                            assert(left_mask_array[(index) * n_object_masks + current_mask_index] != 0);
                            assert(right_mask_array[(index_positive) * n_object_masks + current_mask_index] != 0);
                            assert(left_mask_array[(index_negative) * n_object_masks + current_mask_index] != 0);
                        }
                    }
                }
            }
        }


        std::cout << "[";
        for (int i = 0; i < triplets.size()/3; i++) {
            std::cout  << "[" << triplets[i * 3] << ", " << triplets[i * 3 + 1] << ", " << triplets[i*3+2] << "]";
            if(i + 1 != triplets.size()/3) {
                std::cout << ",";
            }
        }
        std::cout << "]" << std::endl;


        double loss = 0;
        // for each triplet
        int num_triplets = triplets.size() / 3;
//        std::cout << "num_triplets: " << num_triplets << std::endl;
        for (int triplet_num = 0; triplet_num < num_triplets; triplet_num++)
        {
            const int index_i = triplets.at(triplet_num * 3 + 0);
            const int index_j = triplets.at(triplet_num * 3 + 1);
            const int index_k = triplets.at(triplet_num * 3 + 2);
            assert(-1 < index_j);

            // compute the distances
            T D_ij = 0;
            T D_ik = 0;
            for (int c = 0; c < feature_depth; c++)
            {
                int data_i_index = index_i * feature_depth + c;
                int data_j_index = index_j * feature_depth + c;
                int data_k_index = index_k * feature_depth + c;

                if ((0 > data_i_index) || (data_i_index >= feature_arr_size) ||
                    (0 > data_j_index) || (data_j_index >= feature_arr_size) ||
                    (0 > data_k_index) || (data_k_index >= feature_arr_size)) {
                    std::cout << "index out of bounds" << std::endl;
                    std::cout << "\t max index size: " << batch_size * height * width * feature_depth << std::endl;
                    std::cout << "\t data_i_index: " <<  data_i_index << " index_i" << index_i << std::endl;
                    std::cout << "\t data_j_index: " <<  data_j_index << " index_j" << index_j << std::endl;
                    std::cout << "\t data_k_index: " <<  data_k_index << " index_k" << index_k << std::endl;
                    assert(false);
                }

//                std::cout << "hi 2.1" << std::endl;
                D_ij += pow(left_data_array[data_i_index] - right_data_array[data_j_index], 2);
//                std::cout << "hi 2.2" << std::endl;
                D_ik += pow(left_data_array[data_i_index] - left_data_array[data_k_index], 2);
            }
            // add the loss
            double dis = D_ij - D_ik + margin_;
            T old_loss = loss;
            loss += std::max(dis, double(0.0));

            if(num_triplets == 0){
                std::cout << "loss will be nan! (" << loss << ")" << std::endl;
                num_triplets += 1;
            }

            // std::cout << "dis: " << dis << " D_ij: " << D_ij << " D_ik: " << D_ik << std::endl;
            if(old_loss > loss && dis > 0.0 && 1 < 0) {
                std::cout << "loss overflowed, old_loss: " << old_loss << " loss: " << loss << std::endl;
                assert(false);
            }



            // compute gradients
            if (dis > 0) {
                for (int c = 0; c < feature_depth; c++)
                {
                    int diff_i_index = index_i * feature_depth + c;
                    int diff_j_index = index_j * feature_depth + c;
                    int diff_k_index = index_k * feature_depth + c;

                    if ((0 > diff_i_index) || (diff_i_index >= feature_arr_size) ||
                        (0 > diff_j_index) || (diff_j_index >= feature_arr_size) ||
                        (0 > diff_k_index) || (diff_k_index >= feature_arr_size)) {
                        std::cout << "index out of bounds" << std::endl;
                        std::cout << "\t max index size: " << batch_size * height * width * feature_depth << std::endl;
                        std::cout << "\t diff_i_index: " <<  diff_i_index << std::endl;
                        std::cout << "\t diff_j_index: " <<  diff_j_index << std::endl;
                        std::cout << "\t diff_k_index: " <<  diff_k_index << std::endl;
                        assert(false);
                    }

                    // update x_i
//                    std::cout << "hi 2.3, diff_i_index =" << diff_i_index << std::endl;
                    bottom_left_diff[diff_i_index] +=
                            (left_data_array[diff_k_index] - right_data_array[diff_j_index]) / num_triplets;
                    // update x_j
//                    std::cout << "hi 2.4, diff_j_index =" << diff_j_index << std::endl;
                    bottom_right_diff[diff_j_index] +=
                            (right_data_array[diff_j_index] - left_data_array[diff_i_index]) / num_triplets;
                    // update x_k
//                    std::cout << "hi 2.5, diff_k_index =" << diff_k_index << std::endl;
                    bottom_left_diff[diff_k_index] +=
                            (left_data_array[diff_i_index] - left_data_array[diff_k_index]) / num_triplets;
                }
            }
        }
//        std::cout << "pre-scaled loss: " << loss;
        loss /= num_triplets * 2.0;
//        std::cout << " scaled loss: " << loss << std::endl;
//		top_data(0) = T(loss);
        top_data.setConstant(T(loss));
//        std::cout << "hi3" << std::endl;

#ifdef DEBUG_NAN
        feclearexcept(FE_ALL_EXCEPT);
#endif
    }

private:
    float margin_;
    int negative_radius_;
};

REGISTER_KERNEL_BUILDER(Name("TripletFlow").Device(DEVICE_CPU).TypeConstraint<float>("T"), TripletFlowOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("TripletFlow").Device(DEVICE_CPU).TypeConstraint<double>("T"), TripletFlowOp<CPUDevice, double>);


//// GPU implementation for forward pass
//bool TripletForwardLaucher(
//	const float* bottom_data, const float* bottom_label, const int* bottom_prediction,
//	const int batch_size, const int height, const int width, const int channels, const int num_classes,
//	const float margin, float* top_data, float* bottom_diff, const Eigen::GpuDevice& d);
//
//static void TripletKernel(
//	OpKernelContext* context, const Tensor* bottom_data, const Tensor* bottom_label, const Tensor* bottom_prediction,
//	const int batch_size, const int height, const int width, const int channels, const int num_classes,
//	const float margin, const TensorShape& tensor_output_shape, const TensorShape& tensor_output_shape_diff)
//{
//  Tensor* top_data = nullptr;
//  Tensor* bottom_diff = nullptr;
//  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &top_data));
//  OP_REQUIRES_OK(context, context->allocate_output(1, tensor_output_shape_diff, &bottom_diff));
//
//  if (!context->status().ok()) {
//	return;
//  }
//
//  TripletForwardLaucher(
//	bottom_data->flat<float>().data(), bottom_label->flat<float>().data(), bottom_prediction->flat<int>().data(),
//	batch_size, height, width, channels, num_classes, margin,
//	top_data->flat<float>().data(), bottom_diff->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
//}
//
//template <class T>
//class TripletOp<Eigen::GpuDevice, T> : public OpKernel {
// public:
//  typedef Eigen::GpuDevice Device;
//
//  explicit TripletOp(OpKernelConstruction* context) : OpKernel(context) {
//	// Get the margin
//	OP_REQUIRES_OK(context,
//				   context->GetAttr("margin", &margin_));
//	// Check that margin is positive
//	OP_REQUIRES(context, margin_ >= 0,
//				errors::InvalidArgument("Need margin >= 0, got ", margin_));
//  }
//
//  void Compute(OpKernelContext* context) override
//  {
//	// Grab the input tensor
//	const Tensor& bottom_data = context->input(0);
//	const Tensor& bottom_label = context->input(1);
//	const Tensor& bottom_prediction = context->input(2);
//
//	// data should have 4 dimensions.
//	OP_REQUIRES(context, bottom_data.dims() == 4,
//				errors::InvalidArgument("data must be 4-dimensional"));
//
//	OP_REQUIRES(context, bottom_label.dims() == 4,
//				errors::InvalidArgument("label must be 4-dimensional"));
//
//	OP_REQUIRES(context, bottom_prediction.dims() == 3,
//				errors::InvalidArgument("prediction must be 3-dimensional"));
//
//	// batch size
//	int batch_size = bottom_data.dim_size(0);
//	// height
//	int height = bottom_data.dim_size(1);
//	// width
//	int width = bottom_data.dim_size(2);
//	// number of channels
//	int num_channels = bottom_data.dim_size(3);
//	int num_classes = bottom_label.dim_size(3);
//
//	// Create output tensors
//	// loss
//	int dim = 1;
//	TensorShape output_shape;
//	TensorShapeUtils::MakeShape(&dim, 1, &output_shape);
//
//	// bottom diff
//	TensorShape output_shape_diff = bottom_data.shape();
//
//	TripletKernel(context, &bottom_data, &bottom_label, &bottom_prediction, batch_size, height,
//	  width, num_channels, num_classes, margin_, output_shape, output_shape_diff);
//  }
// private:
//  float margin_;
//};
//
//REGISTER_KERNEL_BUILDER(Name("TripletFlow").Device(DEVICE_GPU).TypeConstraint<float>("T"), TripletOp<Eigen::GpuDevice, float>);


// compute gradient
template <class Device, class T>
class TripletFlowGradOp : public OpKernel {
public:
    explicit TripletFlowGradOp(OpKernelConstruction* context) : OpKernel(context) {
        // Get the margin
        OP_REQUIRES_OK(context,
                       context->GetAttr("margin", &margin_));
        // Check that margin is positive
        OP_REQUIRES(context, margin_ >= 0,
                    errors::InvalidArgument("Need margin >= 0, got ", margin_));
    }

    void Compute(OpKernelContext* context) override
    {
        const Tensor& bottom_left_diff = context->input(0);
        auto bottom_left_diff_flat = bottom_left_diff.flat<T>();

        const Tensor& bottom_right_diff = context->input(1);
        auto bottom_right_diff_flat = bottom_right_diff.flat<T>();

        const Tensor& out_backprop = context->input(2);
        T loss = out_backprop.flat<T>()(0);

        // data should have 4 dimensions.
        OP_REQUIRES(context, bottom_left_diff.dims() == 4,
                    errors::InvalidArgument("bottom diff must be 4-dimensional"));

        // data should have 4 dimensions.
        OP_REQUIRES(context, bottom_right_diff.dims() == 4,
                    errors::InvalidArgument("bottom_right_diff must be 4-dimensional"));

        // batch size
        int batch_size = bottom_left_diff.dim_size(0);
        // height
        int height = bottom_left_diff.dim_size(1);
        // width
        int width = bottom_left_diff.dim_size(2);
        // number of channels
        int num_channels = bottom_left_diff.dim_size(3);

//        std::cout << "batch_size: " << batch_size << " height: " << height << " width: " << width << " num_channels: " << num_channels << std::endl;
//        std::cout << "\tout_backprop dims: " << out_backprop.dims() << std::endl;

        // construct the output shape
        TensorShape output_shape = bottom_left_diff.shape();
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
        auto top_data_left = output->template flat<T>();

        OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &output));
        auto top_data_right = output->template flat<T>();

        for (int i = 0; i < batch_size * height * width * num_channels; i++)
        {
            top_data_left(i) = loss * bottom_left_diff_flat(i);
            top_data_right(i) = loss * bottom_right_diff_flat(i);
        }
    }
private:
    float margin_;
};

REGISTER_KERNEL_BUILDER(Name("TripletFlowGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"), TripletFlowGradOp<CPUDevice, float>);
//
//
//bool TripletBackwardLaucher(const float* top_diff, const float* bottom_diff, const int batch_size,
//	const int height, const int width, const int channels, float* output, const Eigen::GpuDevice& d);
//
//static void TripletFlowGradKernel(
//	OpKernelContext* context, const Tensor* bottom_diff, const Tensor* out_backprop,
//	const int batch_size, const int height, const int width, const int channels,
//	const TensorShape& tensor_output_shape)
//{
//  Tensor* output = nullptr;
//  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output));
//
//  if (!context->status().ok()) {
//	return;
//  }
//
//  TripletBackwardLaucher(
//	out_backprop->flat<float>().data(), bottom_diff->flat<float>().data(),
//	batch_size, height, width, channels, output->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
//}


//template <class T>
//class TripletFlowGradOp<Eigen::GpuDevice, T> : public OpKernel {
// public:
//  typedef Eigen::GpuDevice Device;
//
//  explicit TripletFlowGradOp(OpKernelConstruction* context) : OpKernel(context) {
//	// Get the margin
//	OP_REQUIRES_OK(context,
//				   context->GetAttr("margin", &margin_));
//	// Check that margin is positive
//	OP_REQUIRES(context, margin_ >= 0,
//				errors::InvalidArgument("Need margin >= 0, got ", margin_));
//  }
//
//  void Compute(OpKernelContext* context) override
//  {
//	const Tensor& bottom_diff = context->input(0);
//	const Tensor& out_backprop = context->input(1);
//
//	// data should have 4 dimensions.
//	OP_REQUIRES(context, bottom_diff.dims() == 4,
//				errors::InvalidArgument("bottom diff must be 4-dimensional"));
//
//	// batch size
//	int batch_size = bottom_diff.dim_size(0);
//	// height
//	int height = bottom_diff.dim_size(1);
//	// width
//	int width = bottom_diff.dim_size(2);
//	// number of channels
//	int num_channels = bottom_diff.dim_size(3);
//
//	// construct the output shape
//	TensorShape output_shape = bottom_diff.shape();
//
//	// run the kernel
//	TripletFlowGradKernel(
//	  context, &bottom_diff, &out_backprop, batch_size, height, width, num_channels, output_shape);
//  }
// private:
//  float margin_;
//};
//
//REGISTER_KERNEL_BUILDER(Name("TripletFlowGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), TripletFlowGradOp<Eigen::GpuDevice, float>);
