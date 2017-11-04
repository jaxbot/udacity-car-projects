#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  // Accumulate squared residuals.
  VectorXd squared_residuals(4);
  squared_residuals << 0, 0, 0, 0;
  for(int i=0; i < estimations.size(); ++i){
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    squared_residuals += residual;
  }

  // Calculate the mean.
  VectorXd mean(4);
  mean = squared_residuals / estimations.size();

  // Calculate the squared root
  rmse = mean.array().sqrt();

  //return the result
  return rmse;
}
