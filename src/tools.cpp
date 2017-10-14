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

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3,4);
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // Check division by zero.
  if (px + py == 0) {
      cout << "Division by zero." << endl;
      return Hj;
  }

  // Compute the Jacobian matrix.
  float sum_of_squares = pow(px, 2) + pow(py, 2);
  float rss = sqrt(sum_of_squares);
  float croot = sum_of_squares * rss;

  Hj <<
      px / rss, py / rss, 0, 0,
      -py / sum_of_squares, px / sum_of_squares, 0, 0,
      py * (vx * py - vy * px) / croot, px * (vy * px - vx * py) / croot, px / rss, py / rss;

  return Hj;
}
