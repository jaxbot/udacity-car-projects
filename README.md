# CarND-Controls-MPC
Self-Driving Car Engineer Nanodegree Program

## Video

<a href="output/mpc.mp4"><img src="output/mpc.gif" alt="Animation of Udacity simulator car driving down the track, with the MPC prediction line in front of the vehicle."></a>

## Writeup

>Student describes their model in detail. This includes the state, actuators and update equations.

The model is based on the MPC described in the classroom module:

Actuators: steering value (-25 deg to 25 deg), throttle (-1.0 to 1.0)

State: vehicle x, y, and heading (psi), velocity, crosstrack error, and heading error.

The update equations also follow those given the lecture:

```
x_t+1 = x_t + v_t * cos(psi_t) * dt
y_t+1 = y_t + v_t * sin(psi_t) * dt
psi_t+1 = psi_t + v_t / Lf * delta_t * dt // where delta_t is the steering angle at time t, Lf is configuration paren given.
v_t+1 = v_t + a_t * dt // where a_t is the throttle input at time t
cte_t+1 = y_t - f(x_t) + v_t * dt * sin(epsi_t)
epsi_t+1 = psi_t - psides_t + (v_t / Lf * delta_t * dt)

where psides_t = atan(c[1] + 2 * c[2] * x_t + 3 * c[3] * x_t^2) and c is the coefficients from the fit polynomial.
```

>Student discusses the reasoning behind the chosen N (timestep length) and dt (elapsed duration between timesteps) values. Additionally the student details the previous values tried.

I chose N = 8 and dt = 0.1 for my project. The dt value was chosen simply because every timestep in the simulator is delayed by 100ms, due to the added control latency. This means that no control inputs would occur until at least 100ms later, so using 100ms as a timestep was intuitive. The N value was chosen as a compromise between processing time (lower N value = faster frame processing) and the need for a tail long enough to predict sharp turns early (as actuation is delayed by 100ms, braking early is important). 8 was found to be a reasonable middleground, though my previously tried value of 10 also appeared to perform well. No other values for dt were tried, as the simulation would always run with 100ms latency.

>The student implements Model Predictive Control that handles a 100 millisecond latency. Student provides details on how they deal with latency.

Without implementing any specific guards for this, my model that worked perfectly with no added latency would perform poorly at 100ms latency, showing a very distinct oscillation pattern where the model would overcorrect for the CTE and not back off actuation inputs until well after the car passed the CTE on the other side. To work around this, a steep cost was added for actuator use proportional to the steering angle and velocity. This ensured that at high velocities, the actuators would avoid any sharp adjustments that would result in overcorrecting.
