from pid import PID
from yaw_controller import YawController
import math
GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, *args, **kwargs):

        vehicle_mass = kwargs['vehicle_mass']
        fuel_capacity = kwargs['fuel_capacity']
        brake_deadband = kwargs['brake_deadband']
        decel_limit = kwargs['decel_limit']
        accel_limit = kwargs['accel_limit']
        wheel_radius = kwargs['wheel_radius']
        wheel_base = kwargs['wheel_base']
        steer_ratio = kwargs['steer_ratio']
        max_lat_accel = kwargs['max_lat_accel']
        max_steer_angle = kwargs['max_steer_angle']
        min_speed = 0 # TODO: What is a good value for min speed?


        #self.linear_pid = PID(kp=0.2, ki=0.005, kd=0.1, mn=decel_limit, mx=0.5 * accel_limit)
        self.linear_pid = PID(kp=0.8, ki=0, kd=0.05, mn=decel_limit, mx=0.5 * accel_limit)
        self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)
        self.steering_pid = PID(kp=0.15, ki=0.001, kd=0.1, mn=-max_steer_angle, mx=max_steer_angle)

    def reset(self):
        self.linear_pid.reset()
        self.steering_pid.reset()

    def control(self, proposed_linear_velocity, proposed_angular_velocity, current_linear_velocity, cross_track_error, duration_in_seconds):
        linear_velocity_error = proposed_linear_velocity - current_linear_velocity

        velocity_correction = self.linear_pid.step(linear_velocity_error, duration_in_seconds)

        brake = 0
        throttle = velocity_correction

        if(throttle < 0):
            brake = 1000 * abs(throttle)
            throttle = 0

        predictive_steering = self.yaw_controller.get_steering(proposed_linear_velocity, proposed_angular_velocity, current_linear_velocity)
        corrective_steering = self.steering_pid.step(cross_track_error, duration_in_seconds)
        steering = predictive_steering + corrective_steering

        return throttle, brake, steering
