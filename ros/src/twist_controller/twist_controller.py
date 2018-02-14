
import rospy

from yaw_controller import YawController
from pid import PID

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, *args, **kwargs):
        self.yaw_controller = YawController(
                                wheel_base=kwargs['wheel_base'],
                                steer_ratio=kwargs['steer_ratio'],
                                min_speed=kwargs['min_speed'],
                                max_lat_accel=kwargs['max_lat_accel'],
                                max_steer_angle=kwargs['max_steer_angle'])
        self.throttle_pid = PID(kp=0.5, ki=0.05, kd=0.1, mn=0.0, mx=1.0)

    def control(self, *args, **kwargs):
        target_velocity = kwargs['proposed_linear_velocity']
        current_velocity = kwargs['current_linear_velocity']
        dbw_enabled = kwargs['dbw_enabled']

        if dbw_enabled:
            steering = self.yaw_controller.get_steering(
                    linear_velocity=target_velocity,
                    angular_velocity=kwargs['proposed_angular_velocity'],
                    current_velocity=current_velocity)

            error = target_velocity - current_velocity
            throttle = self.throttle_pid.step(error, 0.02)
        else:
            self.throttle_pid.reset()
            throttle = 0.
            steering = 0.

        return throttle, 0., steering
