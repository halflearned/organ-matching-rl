//
//  policy.hpp
//  policy
//
//  Created by Vitor Baisi Hadad on 11/7/17.
//  Copyright © 2017 halflearned. All rights reserved.
//

#ifndef policy_hpp
#define policy_hpp

#include <stdio.h>
#include <iostream>
#include <tiny_dnn/tiny_dnn.h>

using namespace tiny_dnn;
using namespace tiny_dnn::layers;
using namespace tiny_dnn::activation;

class gcn : public layer {
public:
    gcn(size_t x_size, size_t y_size)
    :layer({vector_type::data, vector_type::weight, vector_type::bias}, // x, W and b
           {vector_type::data}),
    x_size_(x_size),
    y_size_(y_size)
    {}
    
std::string layer_type() const override {
    return "graph-convolutional";
}
    
std::vector<shape3d> in_shape() const override {
    // return input shapes
    // order of shapes must be equal to argument of layer constructor
    return { shape3d(x_size_, 1, 1), // x
             shape3d(x_size_, y_size_, 1), // W
             shape3d(y_size_, 1, 1) }; // b
}

std::vector<shape3d> out_shape() const override {
    return { shape3d(y_size_, 1, 1) }; // y
}
    

void forward_propagation(size_t worker_index,
                         const std::vector<vec_t*>& in_data,
                         std::vector<vec_t*>& out_data) {
    const vec_t& x = *in_data[0]; // it's size is in_shapes()[0] (=[x_size_,1,1])
    const vec_t& W = *in_data[1];
    const vec_t& b = *in_data[2];
    vec_t& y = *out_data[0];
    
    std::fill(y.begin(), y.end(), 0.0);
    
    // y = Wx+b
    for (size_t r = 0; r < y_size_; r++) {
        for (size_t c = 0; c < x_size_; c++)
            y[r] += W[r*x_size_+c]*x[c];
        y[r] += b[r];
    }
}


void back_propagation(size_t                index,
                      const std::vector<vec_t*>& in_data,
                      const std::vector<vec_t*>& out_data,
                      std::vector<vec_t*>&       out_grad,
                      std::vector<vec_t*>&       in_grad)  {
    const vec_t& curr_delta = *out_grad[0]; // dE/dy (already calculated in next layer)
    const vec_t& x          = *in_data[0];
    const vec_t& W          = *in_data[1];
    vec_t&       prev_delta = *in_grad[0]; // dE/dx (passed into previous layer)
    vec_t&       dW         = *in_grad[1]; // dE/dW
    vec_t&       db         = *in_grad[2]; // dE/db
    
    // propagate delta to prev-layer
    for (size_t c = 0; c < x_size_; c++)
        for (size_t r = 0; r < y_size_; r++)
            prev_delta[c] += curr_delta[r] * W[r*x_size_+c];
    
    // accumulate weight difference
    for (size_t r = 0; r < y_size_; r++)
        for (size_t c = 0; c < x_size_; c++)
            dW[r*x_size_+c] += curr_delta[r] * x[c];
    
    // accumulate bias difference
    for (size_t r = 0; r < y_size_; r++)
        db[r] += curr_delta[r];
}

private:
    size_t x_size_; // number of input elements
    size_t y_size_; // number of output elements
};




#endif /* policy_hpp */
