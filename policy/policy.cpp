//
//  policy.cpp
//  policy
//
//  Created by Vitor Baisi Hadad on 11/7/17.
//  Copyright © 2017 halflearned. All rights reserved.
//

#include "policy.hpp"
#include <stdio.h>



int main() {
    
    network<sequential> net;
    adagrad opt;
    net << layers::fc(2, 3) << activation::tanh()
    << layers::fc(3, 4) << activation::softmax();
    
    // input_data[0] should be classified to id:3
    // input_data[1] should be classified to id:1
    std::vector<vec_t> input_data    { { 1, 0 }, { 0, 2 } };
    std::vector<label_t> desired_out {        3,        1 };
    size_t batch_size = 1;
    size_t epochs = 30;
    
    net.train<mse>(opt, input_data, desired_out, batch_size, epochs);
    
};
