#include <fstream>
#include <iostream>
#include "kalman_filter.h"
#define PI 3.141592653

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

// Predict the state
void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

// Update the state by using Kalman Filter equations
void KalmanFilter::Update(const VectorXd &z) {
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;

  UpdateInternal(y);
}

// Update the state by using Extended Kalman Filter equations
void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // Map from Cartesian to polar coordinates.
  VectorXd h = VectorXd(3);
  float rho, theta, rhodot;
  rho = sqrt(x_(0) * x_(0) + x_(1) * x_(1));
  theta = atan2(x_(1), x_(0));

  // Prevent division by zero.
  if (fabs(rho) < 0.001) {
    rhodot = 0;
  } else {
    rhodot = (x_(0) * x_(2) + x_(1) * x_(3)) / rho;
  }

  h << rho, theta, rhodot;

  VectorXd y = z - h;

  while (y[1] < -PI) {
    y[1] += 2.0 * PI;
  }
  while (y[1] > PI) {
    y[1] -= 2.0 * PI;
  }

  UpdateInternal(y);
}

// Update the state by using Kalman Filter equations
void KalmanFilter::UpdateInternal(const VectorXd &y) {
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  // New estimate.
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
