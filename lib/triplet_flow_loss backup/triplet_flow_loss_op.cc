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

#define N_LOSS_THREADS 8
#define USE_THREADS false
bool DENSE_SAMPLING = false;
bool DENSE_SAMPLING_orrig = DENSE_SAMPLING;
// n_triplets = n_pixels / TRIPLET_DENSITY
#define MAX_TRIPLETS 200
#define N_NEGATIVES 4

#include <stdio.h>
#include <cfloat>
#include <math.h>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <unistd.h>
#include <thread>
#include <functional>
#include <unistd.h>
#include <csignal>


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


REGISTER_OP("TripletFlow")
.Attr("T: {float}")
.Attr("margin: float")
.Attr("negative_radius: int")
.Input("left_data: T")
.Input("right_data: T")
.Input("gt_flow: T")
.Input("occluded_mask: int32")
.Output("loss: T")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->Scalar());
    return Status::OK();
})
.Output("bottom_left_diff: T")
.Output("bottom_right_diff: T");

REGISTER_OP("TripletFlowGrad")
.Attr("T: {float}")
.Attr("margin: float")
.Input("bottom_left_diff: T")
.Input("bottom_right_diff: T")
.Input("grad: T")
.Output("output_left: T")
.Output("output_right: T");


template <typename T>
class WorkFragment {

public:
    WorkFragment(const T * left_data_array, const T * right_data_array, const T * flow_array, const int* occluded_array)
            : left_data_array(left_data_array), right_data_array(right_data_array), flow_array(flow_array), occluded_array(occluded_array) {}
    ~WorkFragment() {}

    std::thread* this_thread;
    bool done = false;
    int n_triplets_added = 0;
    double individual_loss = 0.0;

    int batch_size = -1;
    int height = -1;
    int width = -1;
    int feature_depth = -1;
    const T *left_data_array;
    const T *right_data_array;
    const T *flow_array;
    const int *occluded_array;
    int n = -1;
    int feature_arr_size = -1;
    int saftey_margin = -1;
    int negative_radius_ = -1;
    int flow_arr_size = -1;

    T *bottom_left_diff;
    T *bottom_right_diff;

    float margin_ = -1;
    int n_approx_triplets = -1;
};


int intRand(const int & min, const int & max) {
    static thread_local std::mt19937 generator;
    std::uniform_int_distribution<int> distribution(min,max);
    return distribution(generator);
}


//template <typename T>
void do_work_unit(int h_start, int h_end, int batch_size, int height, int width, int feature_depth,
                  const float *left_data_array, const float *right_data_array, const float *flow_array, const int *occluded_array,
                  int n, int feature_arr_size, int saftey_margin, int negative_radius_, int flow_arr_size,
                  float *bottom_left_diff, float *bottom_right_diff, float margin_, int n_approx_triplets,
                  std::vector<int>& n_triplets_added, std::vector<float>& individual_loss, int n_triplets) {

    for (int h = h_start; h < h_end; h++)
    {
        for (int w_raw = 0; w_raw < width; w_raw++)
        {
            int w = 0;
            if(DENSE_SAMPLING){
                w = w_raw;
            }
            else {
                w = std::rand() % width;
                if(w_raw >= n_triplets){
                    // don't sample any more with the current h
                    w_raw = width;
                    continue;
                }
            }

            // anchor
            int index = n * height * width + h * width + w;
            if(occluded_array[index] != 0)
            {
//                        std::cout << "point occluded, h: " << h << " w: " << w << std::endl;
                if(!DENSE_SAMPLING && (std::rand() % 10 < 9)) {
                    w_raw--;
                }
                continue;  // don't use this point
            }

            // find the corresponding pixel
            if(index*2+1 >= flow_arr_size) {
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
                continue;  // corresponding right image location is outside image
            }
            int index_positive = n * height * width + h_positive * width + w_positive;

            // sample a negative pixel
            // check the predicted label of this pixel for hard negative

//                    int possible_h = std::rand() % (negative_radius_*2 - saftey_margin*2);
//                    assert(0 <= possible_h < negative_radius_*2 - saftey_margin*2);
//                    int possible_w = std::rand() % (negative_radius_*2 - saftey_margin*2);
//                    assert(0 <= possible_w < negative_radius_*2 - saftey_margin*2);
//
//                    if(possible_h + saftey_margin > h)
//                        possible_h += saftey_margin * 2;
//                    if(possible_w + saftey_margin > w)
//                        possible_w += saftey_margin * 2;
//
//                    int index_negative = ((n) * height + possible_h) * width + possible_w;

            int h_displacement = intRand(0, negative_radius_*2 - saftey_margin) - negative_radius_;
            if(-1 * saftey_margin < h_displacement){
                h_displacement += saftey_margin * 2;
            }
            int h_negative = crop_val(h_displacement + h_positive, 0, height - 1);

            int w_displacement = intRand(0, negative_radius_*2 - saftey_margin) - negative_radius_;
            if(-1 * saftey_margin < w_displacement){
                w_displacement += saftey_margin * 2;
            }
            int w_negative = crop_val(w_displacement + w_positive, 0, height - 1);

            int index_negative = ((n) * height + h_negative) * width + w_negative;

            // store the triplet
//                    triplets.push_back(index);
//                    triplets.push_back(index_positive);
//                    triplets.push_back(index_negative);
            assert(-1 < index && index < batch_size * height * width);
            assert(-1 < index_positive && index_positive < batch_size * height * width);
            assert(-1 < index_negative && index_negative < batch_size * height * width);


            //********************************************************************
            // Pretend that the triplet was added to a vector then removed here
            //********************************************************************
            n_triplets_added[h] += 1;

            const int index_i = index;
            const int index_j = index_positive;
            const int index_k = index_negative;
            assert(-1 < index_j);


            // compute the distances
            float D_ij = 0;
            float D_ik = 0;
            for (int c = 0; c < feature_depth; c++)
            {
                int data_i_index = index_i * feature_depth + c;
                int data_j_index = index_j * feature_depth + c;
                int data_k_index = index_k * feature_depth + c;
		    int arr_size = batch_size * height * width * feature_depth;

                if ((0 > data_i_index) || (data_i_index >= arr_size) ||
                    (0 > data_j_index) || (data_j_index >= arr_size) ||
                    (0 > data_k_index) || (data_k_index >= arr_size)) {
                    std::cout << "index out of bounds" << std::endl;
                    std::cout << "\t max index size: " << arr_size << std::endl;
                    std::cout << "\t data_i_index: " <<  data_i_index << " index_i " << index_i << std::endl;
                    std::cout << "\t data_j_index: " <<  data_j_index << " index_j " << index_j << std::endl;
                    std::cout << "\t data_k_index: " <<  data_k_index << " index_k " << index_k << std::endl;
                    assert(false);
                }

//                std::cout << "hi 2.1" << std::endl;
                D_ij += pow(left_data_array[data_i_index] - right_data_array[data_j_index], 2);
//                std::cout << "hi 2.2" << std::endl;
                D_ik += pow(left_data_array[data_i_index] - left_data_array[data_k_index], 2);
            }
            // add the loss
            double dis = D_ij - D_ik + margin_;
            double loss_increment = std::max(dis, double(0.0));
            individual_loss[h] += loss_increment;

            // std::cout << "dis: " << dis << " D_ij: " << D_ij << " D_ik: " << D_ik << std::endl;

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
                            (left_data_array[diff_k_index] - right_data_array[diff_j_index]);
                    // update x_j
//                    std::cout << "hi 2.4, diff_j_index =" << diff_j_index << std::endl;
                    bottom_right_diff[diff_j_index] +=
                            (right_data_array[diff_j_index] - left_data_array[diff_i_index]);
                    // update x_k
//                    std::cout << "hi 2.5, diff_k_index =" << diff_k_index << std::endl;
                    bottom_left_diff[diff_k_index] +=
                            (left_data_array[diff_i_index] - left_data_array[diff_k_index]);
                }
            }

        }
    }
//    std::cout << "found " << n_triplets_added << " valid triplets" << std::endl;
}


template <typename T>
void call_from_thread(int id, WorkFragment<T>* fragment_pointer, int h_start, int h_end) {
    std::cout << "Hello, World from " << id<< std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "\tSecond Hello, World from " << id << std::endl;
}


template <typename Device, typename T = float>
class TripletFlowOp : public OpKernel {
public:


    explicit TripletFlowOp(OpKernelConstruction* context) : OpKernel(context) {
        // Get the margin
        OP_REQUIRES_OK(context,
                       context->GetAttr("margin", &margin_));
        // Check that margin is positive
        OP_REQUIRES(context, margin_ >= 0,
                    errors::InvalidArgument("Need margin >= 0, got ", margin_));
        // Get the negative radius
        OP_REQUIRES_OK(context,
                       context->GetAttr("negative_radius", &negative_radius_));
        // Check that negative radius is positive
        OP_REQUIRES(context, negative_radius_ >= 0,
                    errors::InvalidArgument("Need negative radius >= 0, got ", negative_radius_));
    }


    // bottom_data: (batch_size, height, width, channels)
    // bottom_label: (batch_size, height, width, num_classes)
    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor &left_data = context->input(0);
        const T *left_data_array = left_data.flat<T>().data();

        const Tensor &right_data = context->input(1);
        const T *right_data_array = right_data.flat<T>().data();

        const Tensor &flow_tensor = context->input(2);
        const T *flow_array = flow_tensor.flat<T>().data();

        const Tensor &occluded = context->input(3);
        const int *occluded_array = occluded.flat<int>().data();

        // data should have 4 dimensions.
        OP_REQUIRES(context, left_data.dims() == 4,
                    errors::InvalidArgument("left data must be 3-dimensional"));

        OP_REQUIRES(context, right_data.dims() == 4,
                    errors::InvalidArgument("right data must be 3-dimensional"));

        OP_REQUIRES(context, flow_tensor.dims() == 4,
                    errors::InvalidArgument("flow must be 3-dimensional"));

        OP_REQUIRES(context, occluded.dims() == 4,
                    errors::InvalidArgument("occlusions must be 3-dimensional"));

        // batch size
        int batch_size = left_data.dim_size(0);
        // height
        int height = left_data.dim_size(1);
        // width
        int width = left_data.dim_size(2);
        // number of channels
        int feature_depth = left_data.dim_size(3);

//        std::cout << "Image batch_size: " << batch_size << " height: " << height << " width: " << width << " feature_depth: " << feature_depth << std::endl;
//        std::cout << "flow batch_size: " << flow_tensor.dim_size(0) << " height: " << flow_tensor.dim_size(1) << " width: " << flow_tensor.dim_size(2) << " feature_depth: " << flow_tensor.dim_size(3) << std::endl;
//        std::cout << "occluded batch_size: " << occluded.dim_size(0) << " height: " << occluded.dim_size(1) << " width: " << occluded.dim_size(2) << std::endl;

        assert(batch_size == flow_tensor.dim_size(0));
        assert(batch_size == occluded.dim_size(0));
        assert(height == flow_tensor.dim_size(1));
        assert(height == occluded.dim_size(1));
        assert(width == flow_tensor.dim_size(2));
        assert(width == occluded.dim_size(2));

        int feature_arr_size = batch_size * height * width * feature_depth;

        int flow_arr_size =
                flow_tensor.dim_size(0) * flow_tensor.dim_size(1) * flow_tensor.dim_size(2) * flow_tensor.dim_size(3);

        // Create output loss tensor
        int dim = 1;
        TensorShape output_shape;
//		TensorShapeUtils::MakeShape(&dim, 1, &output_shape);

        Tensor *top_data_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &top_data_tensor));
        auto top_data = top_data_tensor->template flat<T>();

        // bottom diff
        TensorShape output_shape_left_diff = left_data.shape();
        Tensor *bottom_left_diff_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, output_shape_left_diff, &bottom_left_diff_tensor));
        T *bottom_left_diff = bottom_left_diff_tensor->template flat<T>().data();
        memset(bottom_left_diff, 0, batch_size * height * width * feature_depth * sizeof(T));

        TensorShape output_shape_right_diff = left_data.shape();
        Tensor *bottom_right_diff_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(2, output_shape_right_diff, &bottom_right_diff_tensor));
        T *bottom_right_diff = bottom_right_diff_tensor->template flat<T>().data();
        memset(bottom_right_diff, 0, batch_size * height * width * feature_depth * sizeof(T));

        // sample triplets to define the loss
        // compute label indexes

//		// classes in the batch

        int max_triplets_per_img = std::min(height * width, MAX_TRIPLETS);

        int num_triplets = 0;
        double loss = 0;
        int saftey_margin = 2;  // the radius around the target where we shouldn't sample
        // sampling
        std::srand(unsigned(std::time(0)));
//        std::vector<int> triplets;  // don't initialize it with a size because some points might be occluded
//        triplets.reserve(batch_size * height * width * 3);

        int n_approx_triplets = 0;
        if(DENSE_SAMPLING)
            n_approx_triplets = batch_size * height * width;
        else
            n_approx_triplets = batch_size * std::min(height * width, MAX_TRIPLETS);

        while(num_triplets == 0) {
            for (int n = 0; n < batch_size; n++) {

                std::vector <std::thread> thread_vect;
                std::vector<int> triplets_added(height, 0);
                std::vector<float> partial_loss(height, 0.0);

                if (USE_THREADS) {
                    int count = height / N_LOSS_THREADS;
                    while (count < height) {
                        int increment = 40;
                        if (count + increment >= height) {
                            increment = height - count - 1;
                        }
                        if (increment != 0) {

                            auto func = std::bind(do_work_unit, count, count + increment, batch_size, height, width,
                                                  feature_depth, (const float *) left_data_array,
                                                  (const float *) right_data_array, (const float *) flow_array,
                                                  (const int *) occluded_array,
                                                  n, feature_arr_size, saftey_margin, negative_radius_, flow_arr_size,
                                                  (float *) bottom_left_diff, (float *) bottom_right_diff, margin_,
                                                  n_approx_triplets, std::ref(triplets_added), std::ref(partial_loss), 0);
                            count += increment;


                            if (thread_vect.size() < N_LOSS_THREADS) {
                                thread_vect.push_back(std::thread(func));
                            } else {
                                thread_vect.at(count % thread_vect.size()).join();
                                thread_vect.at(count % thread_vect.size()) = std::thread(func);
                            }

    //                auto func = std::bind(&WorkFragment<T>::do_work_unit, next_fragment, std::_1);
    //                std::thread t1(do_work_unit<T>, next_fragment, count, count+1);
    //                call_from_thread(count, next_fragment, count, count+1);
    //                auto func = std::bind(do_work_unit<T>, next_fragment, count, count + 1);
    //                std::thread t1(func);
    //		        t1.join();
    //                void do_work_unit(count, count + 1, batch_size, height, width, feature_depth, left_data_array, right_data_array, flow_array, occluded_array, n, feature_arr_size, saftey_margin, negative_radius_, flow_arr_size, bottom_left_diff, bottom_right_diff, margin_, n_approx_triplets, triplets_added, partial_loss) {

    //                do_work_unit(count, count + 1, batch_size, height, width, feature_depth, (const float *) left_data_array,
    //                             (const float *) right_data_array, (const float *) flow_array, (const int *) occluded_array,
    //                             n, feature_arr_size, saftey_margin, negative_radius_, flow_arr_size,
    //                             (float *) bottom_left_diff, (float *) bottom_right_diff, margin_, n_approx_triplets, triplets_added, partial_loss);
    //                loss += partial_loss;
    //                num_triplets += triplets_added;
    //                std::raise(SIGINT);
                        } else {
                            count += 1;  // (end the loop)
                        }
                    }
                    for (int i = 0; i < thread_vect.size(); i++) {
                        thread_vect.at(i).join();
                    }
                } else {
                    if (!DENSE_SAMPLING) {
                        int triplets_per_line = 5;
                        for (int i = 0; i < max_triplets_per_img / triplets_per_line; i++) {
                            int rand_h = std::rand() % height;
                            do_work_unit(rand_h, rand_h + 1, batch_size, height, width,
                                         feature_depth, (const float *) left_data_array,
                                         (const float *) right_data_array, (const float *) flow_array,
                                         (const int *) occluded_array,
                                         n, feature_arr_size, saftey_margin, negative_radius_, flow_arr_size,
                                         (float *) bottom_left_diff, (float *) bottom_right_diff, margin_,
                                         n_approx_triplets, std::ref(triplets_added), std::ref(partial_loss),
                                         triplets_per_line);
                        }
                    } else {
                        for (int i = 0; i < height; i++) {
                            do_work_unit(i, i + 1, batch_size, height, width,
                                         feature_depth, (const float *) left_data_array,
                                         (const float *) right_data_array, (const float *) flow_array,
                                         (const int *) occluded_array,
                                         n, feature_arr_size, saftey_margin, negative_radius_, flow_arr_size,
                                         (float *) bottom_left_diff, (float *) bottom_right_diff, margin_,
                                         n_approx_triplets, std::ref(triplets_added), std::ref(partial_loss),
                                         100000);
                        }
                    }
                }

                for (int i = 0; i < height; i++) {
                    num_triplets += triplets_added[i];
                    loss += partial_loss[i];
                }
            }

            if(num_triplets == 0) {
//                raise(SIGUSR1);
                DENSE_SAMPLING = true;
            }
        }

        DENSE_SAMPLING = DENSE_SAMPLING_orrig;

//        std::cout << "hi2" << std::endl;

        // for each triplet
//        int num_triplets = triplets.size() / 3;
////        std::cout << "num_triplets: " << num_triplets << std::endl;
//        for (int triplet_num = 0; triplet_num < num_triplets; triplet_num++)
//        {
//
//        }
//        std::cout << "pre-scaled loss: " << loss;

        for(int i = 0; i < feature_arr_size; i++){
            bottom_left_diff[i] /= num_triplets;
            bottom_right_diff[i] /= num_triplets;
        }

        if (num_triplets == 0) {
            std::cout << "loss would be nan! (" << loss << ") (no triplets )" << std::endl;
            loss = 0.3;
        }

        loss /= num_triplets * 2.0;

//        std::cout << " scaled loss: " << loss << std::endl;
//		top_data(0) = T(loss);
        top_data.setConstant(T(loss));
//        std::cout << "hi3" << std::endl;
    }

private:
    float margin_;
    int negative_radius_;
};

REGISTER_KERNEL_BUILDER(Name("TripletFlow").Device(DEVICE_CPU).TypeConstraint<float>("T"), TripletFlowOp<CPUDevice, float>);
//REGISTER_KERNEL_BUILDER(Name("TripletFlow").Device(DEVICE_CPU).TypeConstraint<double>("T"), TripletFlowOp<CPUDevice, double>);


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
        Tensor* output_l = nullptr;
        Tensor* output_r = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_l));
        auto top_data_left = output_l->template flat<T>();

        OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &output_r));
        auto top_data_right = output_r->template flat<T>();

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





// GPU implementation for forward pass
bool TripletForwardLaucher(
        const float* left_data, const float* right_data, const float* flow_tensor, const int* occluded_tensor,
        const int batch_size, const int height, const int width, const int channels, const int n_triplets,
        const float margin, const int negative_radius_, float* top_data, float* bottom_left_diff, float* bottom_right_diff, const Eigen::GpuDevice& d);

static void TripletKernel(
	OpKernelContext* context, const Tensor* left_data, const Tensor* right_data, const Tensor* flow_tensor, const Tensor* occluded_tensor,
	const int batch_size, const int height, const int width, const int channels, const int n_triplets,
	const float margin, const int negative_radius, const TensorShape& tensor_output_shape, const TensorShape& tensor_output_shape_diff)
{
    Tensor* top_data = nullptr;
    Tensor* bottom_left_diff = nullptr;
    Tensor* bottom_right_diff = nullptr;

    TensorShape output_shape;
//		TensorShapeUtils::MakeShape(&dim, 1, &output_shape);

    OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &top_data));
    OP_REQUIRES_OK(context, context->allocate_output(1, tensor_output_shape_diff, &bottom_left_diff));
    OP_REQUIRES_OK(context, context->allocate_output(2, tensor_output_shape_diff, &bottom_right_diff));

    if (!context->status().ok()) {
        return;
    }

    TripletForwardLaucher(left_data->flat<float>().data(), right_data->flat<float>().data(),
                          flow_tensor->flat<float>().data(), occluded_tensor->flat<int>().data(),
                          batch_size, height, width, channels, n_triplets, margin, negative_radius,
                          top_data->flat<float>().data(), bottom_left_diff->flat<float>().data(),
                          bottom_right_diff->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
}

template <class T>
class TripletFlowOp<Eigen::GpuDevice, T> : public OpKernel {
public:
    typedef Eigen::GpuDevice Device;

    explicit TripletFlowOp(OpKernelConstruction* context) : OpKernel(context) {
        // Get the margin
        OP_REQUIRES_OK(context,
                   context->GetAttr("margin", &margin_));
        // Check that margin is positive
        OP_REQUIRES(context, margin_ >= 0,
                errors::InvalidArgument("Need margin >= 0, got ", margin_));
        // Get the negative radius
        OP_REQUIRES_OK(context,
                     context->GetAttr("negative_radius", &negative_radius_));
        // Check that negative radius is positive
        OP_REQUIRES(context, negative_radius_ >= 0,
                  errors::InvalidArgument("Need negative radius >= 0, got ", negative_radius_));
    }

    void Compute(OpKernelContext* context) override
    {

        const Tensor &left_data = context->input(0);
        const Tensor &right_data = context->input(1);
        const Tensor &flow_tensor = context->input(2);
        const Tensor &occluded_tensor = context->input(3);

        // data should have 4 dimensions.
        OP_REQUIRES(context, left_data.dims() == 4,
                  errors::InvalidArgument("left data must be 3-dimensional"));

        OP_REQUIRES(context, right_data.dims() == 4,
                  errors::InvalidArgument("right data must be 3-dimensional"));

        OP_REQUIRES(context, flow_tensor.dims() == 4,
                  errors::InvalidArgument("flow must be 3-dimensional"));

        OP_REQUIRES(context, occluded_tensor.dims() == 4,
                  errors::InvalidArgument("occlusions must be 3-dimensional"));


        // batch size
        int batch_size = left_data.dim_size(0);
        // height
        int height = left_data.dim_size(1);
        // width
        int width = left_data.dim_size(2);
        // number of channels
        int num_channels = left_data.dim_size(3);


	    // Create output tensors
        // loss
        int dim = 1;
        TensorShape output_shape;
//        TensorShapeUtils::MakeShape(&dim, 1, &output_shape);
//        output_shape.clear();

        // bottom diff
        TensorShape output_diff_shape = left_data.shape();

        TripletKernel(context, &left_data, &right_data, &flow_tensor, &occluded_tensor, batch_size, height,
        width, num_channels, MAX_TRIPLETS, margin_, negative_radius_, output_shape, output_diff_shape);
    }
private:
    float margin_;
    int negative_radius_;
};

//REGISTER_KERNEL_BUILDER(Name("TripletFlow").Device(DEVICE_GPU).TypeConstraint<float>("T"), TripletFlowOp<Eigen::GpuDevice, float>);



bool TripletFlowBackwardLaucher(const float* top_diff, const float* bottom_left_diff, const float* bottom_right_diff, const int batch_size,
	const int height, const int width, const int channels, float* output_l, float* output_r, const Eigen::GpuDevice& d);

static void TripletFlowGradKernel(
	OpKernelContext* context, const Tensor* bottom_left_diff, const Tensor* bottom_right_diff, const Tensor* out_backprop,
	const int batch_size, const int height, const int width, const int channels,
	const TensorShape& tensor_output_shape)
{
    std::cout << "allocating gradient output" << std::endl;
    Tensor* output_l = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output_l));
    Tensor* output_r = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, tensor_output_shape, &output_r));

  if (!context->status().ok()) {
	return;
  }

  TripletFlowBackwardLaucher(
	out_backprop->flat<float>().data(), bottom_left_diff->flat<float>().data(), bottom_right_diff->flat<float>().data(),
	batch_size, height, width, channels, output_l->flat<float>().data(), output_r->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
}


template <class T>
class TripletFlowGradOp<Eigen::GpuDevice, T> : public OpKernel {
 public:
  typedef Eigen::GpuDevice Device;

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
      std::cout << "starting to compute gradient" << std::endl;
    const Tensor& bottom_left_diff = context->input(0);
    const Tensor& bottom_right_diff = context->input(1);
    const Tensor& out_backprop = context->input(2);

	// data should have 4 dimensions.
	OP_REQUIRES(context, bottom_left_diff.dims() == 4,
				errors::InvalidArgument("bottom diff must be 4-dimensional"));

	// batch size
	int batch_size = bottom_left_diff.dim_size(0);
	// height
	int height = bottom_left_diff.dim_size(1);
	// width
	int width = bottom_left_diff.dim_size(2);
	// number of channels
	int num_channels = bottom_left_diff.dim_size(3);

	// construct the output shape
	TensorShape output_shape = bottom_left_diff.shape();

	// run the kernel
	TripletFlowGradKernel(
	  context, &bottom_left_diff, &bottom_right_diff, &out_backprop, batch_size, height, width, num_channels, output_shape);
  }
 private:
  float margin_;
};

//REGISTER_KERNEL_BUILDER(Name("TripletFlowGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), TripletFlowGradOp<Eigen::GpuDevice, float>);
