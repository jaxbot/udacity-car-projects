#include "PID.h"

using namespace std;

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
    this->Kp = Kp;
    this->Ki = Ki;
    this->Kd = Kd;

    p_error = 0;
    i_error = 0;
    d_error = 0;
}

void PID::UpdateError(double cte) {
    // Set p_error here if unset so that d_error can be initized on first frame.
    if (p_error == 0) {
        p_error = cte;
    }

    // Update the error based on equation given in module.
    d_error = cte - p_error;
    p_error = cte;

    i_error += cte;
}

double PID::TotalError() {
    return -Kp * p_error - Kd * d_error - Ki * i_error;
}

