#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * Much of this code is based on the Unscented Kalman Filters module in Udacity's Car Nanodegree program.
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.3;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // State dimension.
  n_x_ = 5;

  // Augmented state dimension.
  n_aug_ = 7;

  // Sigma point spreading parameter (using 3 - n_aug per module recommendation)
  lambda_ = 3 - n_aug_;

  // Predicted sigma points matrix.
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Calculate weights
  weights_ = VectorXd(2 * n_aug_ + 1);
  for (int i = 0; i < 2*n_aug_+1; i++) {
      if (i == 0) {
          weights_(i) = lambda_ / (lambda_ + n_aug_);
      } else {
          weights_(i) = 1 / (2 * (lambda_ + n_aug_));
      }
  }

  // Initialize state vector.
  x_ << 1, 1, 1, 1, 0.1;

  // Initialize covariance matrix.
  P_ <<
    0.2, 0, 0, 0, 0,
    0, 0.2, 0, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 0, 1, 0,
    0, 0, 0, 0, 1;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float rho = meas_package.raw_measurements_(0);
      float phi = meas_package.raw_measurements_(1);
      float rhodot = meas_package.raw_measurements_(2);
      float px = rho * cos(phi);
      float py = rho * sin(phi);
      x_(0) = px;
      x_(1) = py;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_(0) = meas_package.raw_measurements_(0);
      x_(1) = meas_package.raw_measurements_(1);
    }

    is_initialized_ = true;
    time_us_ = meas_package.timestamp_;
  } else {
    float dt = meas_package.timestamp_ - time_us_;
    // Convert microseconds to seconds.
    dt = dt / 1000000.0;
    time_us_ = meas_package.timestamp_;

    Prediction(dt);

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      UpdateRadar(meas_package);
    } else {
      UpdateLidar(meas_package);
    }
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  // Create augmented mean vector.
  VectorXd x_aug = VectorXd(n_aug_);

  // Create augmented state covariance.
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  // Create sigma point matrix.
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  x_aug.head(5) = x_;
  // Set 6th and 7th elements (nu_a and nu_phidot) to 0.
  x_aug(5) = 0;
  x_aug(6) = 0;

  // Create augmented covariance matrix.
  P_aug.fill(0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;

  // Create square root matrix.
  MatrixXd sqrt_P_aug = P_aug.llt().matrixL();

  // Create augmented sigma points.
  Xsig_aug.col(0) = x_aug;
  for (int i = 1; i < n_aug_ + 1; i++) {
      Xsig_aug.col(i) = x_aug + sqrt(lambda_ + n_aug_) * sqrt_P_aug.col(i - 1);
      Xsig_aug.col(i + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * sqrt_P_aug.col(i - 1);
  }

  // Predict sigma points.
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    float px = Xsig_aug(0, i);
    float py = Xsig_aug(1, i);
    float v = Xsig_aug(2, i);
    float phi = Xsig_aug(3, i);
    float phidot = Xsig_aug(4, i);
    float nu = Xsig_aug(5, i);
    float nudot = Xsig_aug(6, i);

    float predicted_x, predicted_y;

    // Avoid division by zero.
    if (fabs(phidot) > 0.01) {
        predicted_x = px + v / phidot * (sin(phi + phidot * delta_t) - sin(phi));
        predicted_y = py + v / phidot * (cos(phi) - cos(phi + phidot * delta_t));
    } else {
        predicted_x = px + v * delta_t * cos(phi);
        predicted_y = py + v * delta_t * sin(phi);
    }

    float predicted_v = v + nu * delta_t;
    float predicted_phi = phi + phidot * delta_t + 0.5 * nudot * pow(delta_t, 2);
    float predicted_phidot = phidot + nudot * delta_t;

    predicted_x += 0.5 * nu * pow(delta_t, 2) * cos(phi);
    predicted_y += 0.5 * nu * pow(delta_t, 2) * sin(phi);

    // Write predicted sigma points into right column.
    Xsig_pred_(0, i) = predicted_x;
    Xsig_pred_(1, i) = predicted_y;
    Xsig_pred_(2, i) = predicted_v;
    Xsig_pred_(3, i) = predicted_phi;
    Xsig_pred_(4, i) = predicted_phidot;
  }

  // Predict state mean.
  x_.fill(0);
  for (int i = 0; i < 2*n_aug_+1; i++) {
      x_ += weights_(i) * Xsig_pred_.col(i);
  }

  // Predict state covariance matrix.
  P_.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
      VectorXd delta = Xsig_pred_.col(i) - x_;

      while (delta(3) >  M_PI) delta(3) -= 2.0 * M_PI;
      while (delta(3) < -M_PI) delta(3) += 2.0 * M_PI;

      P_ = P_ + weights_(i) * delta * delta.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  VectorXd z = meas_package.raw_measurements_;
  int sigma_points = 2 * n_aug_ + 1;
  MatrixXd Zsig = MatrixXd(2, sigma_points);
  Zsig.fill(0);

  // Transform sigma points into measurement space.
  for (int i = 0; i < sigma_points; i++) {
      float px = Xsig_pred_(0, i);
      float py = Xsig_pred_(1, i);

      Zsig(0, i) = px;
      Zsig(1, i) = py;
  }

  // Calculate mean predicted measurement.
  VectorXd z_pred = VectorXd(2);
  z_pred.fill(0);
  for (int i = 0; i < sigma_points; i++) {
      z_pred += weights_(i) * Zsig.col(i);
  }

  // Calculate measurement covariance matrix S.
  MatrixXd S = MatrixXd(2, 2);
  S.fill(0);
  for (int i = 0; i < sigma_points; i++) {
      VectorXd delta = Zsig.col(i) - z_pred;

      S += weights_(i) * delta * delta.transpose();
  }

  MatrixXd R = MatrixXd(2, 2);
  R <<
      pow(std_laspx_, 2), 0,
      0, pow(std_laspy_, 2);
  S += R;

  MatrixXd Tc = MatrixXd(n_x_, 2);

  // Calculate cross correlation matrix.
  Tc.fill(0);
  for (int i = 0; i < sigma_points; i++) {
      VectorXd x_delta = Xsig_pred_.col(i) - x_;
      VectorXd z_delta = Zsig.col(i) - z_pred;

      Tc += weights_(i) * x_delta * z_delta.transpose();
  }

  // Calculate Kalman gain K.
  MatrixXd K = Tc * S.inverse();
  VectorXd z_delta = z - z_pred;

  x_ += K * z_delta;
  P_ -= K * S * K.transpose();
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  VectorXd z = meas_package.raw_measurements_;

  int sigma_points = 2 * n_aug_ + 1;
  MatrixXd Zsig = MatrixXd(3, sigma_points);

  // Transform sigma points into measurement space.
  for (int i = 0; i < sigma_points; i++) {
      float px = Xsig_pred_(0, i);
      float py = Xsig_pred_(1, i);
      float v = Xsig_pred_(2, i);
      float yaw = Xsig_pred_(3, i);
      float v1 = cos(yaw) * v;
      float v2 = sin(yaw) * v;

      Zsig(0, i) = sqrt(pow(px, 2) + pow(py, 2));
      Zsig(1, i) = atan2(py, px);
      Zsig(2, i) = (v1 * px + v2 * py) / sqrt(pow(px, 2) + pow(py, 2));
  }

  // Calculate mean predicted measurement.
  VectorXd z_pred = VectorXd(3);
  z_pred.fill(0);
  for (int i = 0; i < sigma_points; i++) {
      z_pred += weights_(i) * Zsig.col(i);
  }

  // Calculate measurement covariance matrix S.

  MatrixXd S = MatrixXd(3, 3);
  S.fill(0);
  for (int i = 0; i < sigma_points; i++) {
      VectorXd delta = Zsig.col(i) - z_pred;

      while (delta(1) >  M_PI) delta(1) -= 2.0 * M_PI;
      while (delta(1) < -M_PI) delta(1) += 2.0 * M_PI;

      S += weights_(i) * delta * delta.transpose();
  }

  MatrixXd R = MatrixXd(3, 3);
  R <<
      pow(std_radr_, 2), 0, 0,
      0, pow(std_radphi_, 2), 0,
      0, 0, pow(std_radrd_, 2);
  S += R;

  MatrixXd Tc = MatrixXd(n_x_, 3);
  // Calculate cross correlation matrix.
  Tc.fill(0);
  for (int i = 0; i < sigma_points; i++) {
      VectorXd x_delta = Xsig_pred_.col(i) - x_;
      VectorXd z_delta = Zsig.col(i) - z_pred;

      while (x_delta(3) >  M_PI) x_delta(3) -= 2.0 * M_PI;
      while (x_delta(3) < -M_PI) x_delta(3) += 2.0 * M_PI;

      while (z_delta(1) >  M_PI) z_delta(1) -= 2.0 * M_PI;
      while (z_delta(1) < -M_PI) z_delta(1) += 2.0 * M_PI;

      Tc += weights_(i) * x_delta * z_delta.transpose();
  }

  // Calculate Kalman gain K.
  MatrixXd K = Tc * S.inverse();
  VectorXd z_delta = z - z_pred;
  while (z_delta(1) >  M_PI) z_delta(1) -= 2.0 * M_PI;
  while (z_delta(1) < -M_PI) z_delta(1) += 2.0 * M_PI;

  x_ += K * z_delta;
  P_ -= K * S * K.transpose();
}
