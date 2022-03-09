// This file contains the diff qp

#include <torch/torch.h>
#include <eigen3/Eigen/Dense>
#include <iostream>

using namespace torch::autograd;

namespace diff_qp{

    class DiffQP : Function<DiffQP> {

        public:
            
            static torch::Tensor forward(
                        AutogradContext *ctx, 
                        torch::Tensor Q, 
                        torch::Tensor q, 
                        Eigen::MatrixXd G,
                        Eigen::VectorXd h,
                        Eigen::MatrixXd A,
                        Eigen::VectorXd b);

            static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs);

    };
}

