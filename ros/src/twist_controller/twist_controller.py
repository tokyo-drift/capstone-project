from pid import PID

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

        self.linear_pid = PID(0.12, 0.0005, 0.04, -accel_limit, accel_limit)
    

    def control(self, proposed_linear_velocity, proposed_angular_velocity, current_linear_velocity):
        linear_velocity_error = proposed_linear_velocity - current_linear_velocity

        # TODO: what exactly is sample time?
        sample_time = 0.05

        velocity_correction = self.linear_pid.step(linear_velocity_error, sample_time)

        throttle = velocity_correction

        if(throttle < 0):
            brake = 1000 * abs(throttle)
            throttle = 0.

        # TODO: Implement angular velocity for steering
        return throttle, brake, 0.
