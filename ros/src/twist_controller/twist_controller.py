
import rospy

from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, *args, **kwargs):
        # Capture input variables
        self.vehicle_mass = kwargs['vehicle_mass']
        self.fuel_capacity = kwargs['fuel_capacity']
        self.wheel_radius = kwargs['wheel_radius']
        self.decel_limit = kwargs['decel_limit']
        # Initialize the controller
        self.yaw_controller = YawController(
                                wheel_base=kwargs['wheel_base'],
                                steer_ratio=kwargs['steer_ratio'],
                                min_speed=kwargs['min_speed'],
                                max_lat_accel=kwargs['max_lat_accel'],
                                max_steer_angle=kwargs['max_steer_angle'])
        # PIDs:
        self.accel_pid = PID(kp=0.5, ki=0.0, kd=0.1,
                mx=kwargs['accel_limit'], mn=kwargs['decel_limit'])
        self.steer_pid = PID(kp=0.5, ki=0.5, kd=0.1,
                mx=kwargs['max_steer_angle'],
                mn=-1.0*kwargs['max_steer_angle'])
        # Filters:
        self.accel_filt = LowPassFilter(tau=10, ts=1)
        self.steer_filt = LowPassFilter(tau=5, ts=1)
        self.brake_filt = LowPassFilter(tau=1.5, ts=1)
        # Set the accel_filter's initial value to 0
        self.accel_filt.filt(0)


    def control(self, *args, **kwargs):
        target_velocity = kwargs['proposed_linear_velocity']
        current_velocity = kwargs['current_linear_velocity']
        dbw_enabled = kwargs['dbw_enabled']
        duration = kwargs['duration']

        if dbw_enabled:
            steer_control = self.yaw_controller.get_steering(
                    linear_velocity=target_velocity,
                    angular_velocity=kwargs['proposed_angular_velocity'],
                    current_velocity=current_velocity)
            # May be run a PID with cross track error?
            steer = self.steer_filt.filt(steer_control)
            # Get throttle output
            vel_error = target_velocity - current_velocity
            accel_control = self.accel_pid.step(vel_error, duration)
            accel = self.accel_filt.filt(accel_control)
            # Get the brake output
            if accel < 0: # Trigger for deceleration only
                brake_control = self.brake_value(accel)
                brake = self.brake_filt.filt(brake_control)
                accel = 0.
            else:
                brake = 0.
        else:
            self.accel_pid.reset()
            self.steer_pid.reset()
            accel = 0.
            steer = 0.
            brake = 0.

        return accel, brake, steer

    def brake_value(self, accel):
        '''
        Calculate torque value for the brake
        @param accel: acceleration input
        '''
        total_mass = self.vehicle_mass + self.fuel_capacity * GAS_DENSITY
        torque = -1.0 * accel * total_mass * self.wheel_radius
        return torque
