from yaw_controller import YawController
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
  def __init__(self, vehicle_mass,
                     fuel_capacity,
                     brake_deadband,
                     decel_limit,
                     accel_limit,
                     wheel_radius,
                     wheel_base,
                     steer_ratio,
                     max_lat_accel,
                     max_steer_angle):

    self.yaw_controller_ = YawController(wheel_base,
                                         steer_ratio,
                                         0.1,
                                         max_lat_accel,
                                         max_steer_angle)

    kp = 0.3
    ki = 0.1
    kd = 0.0
    mn = 0.0
    mx = 0.2

    self.throttle_controller_ = PID(kp, ki, kd, mn, mx)

    tau = 0.5
    ts = 0.02
    self.vel_lpf_ = LowPassFilter(tau, ts)

    self.vehicle_mass_   = vehicle_mass
    self.fuel_capacity_  = fuel_capacity
    self.brake_deadband_ = brake_deadband
    self.decel_limit_    = decel_limit
    self.accel_limit_    = accel_limit
    self.wheel_radius_   = wheel_radius

    self.last_vel_  = 0.0
    self.last_time_ = rospy.get_time()

    pass

  def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):

    if not dbw_enabled:
      self.throttle_controller_.reset()
      return 0.0, 0.0, 0.0

    current_vel = self.vel_lpf_.filt(current_vel)
    steering = self.yaw_controller_.get_steering(linear_vel, angular_vel, current_vel)

    vel_error = linear_vel - current_vel
    self.last_vel_ = current_vel

    current_time = rospy.get_time()
    sample_time = current_time - self.last_time_
    self.last_time_ = current_time

    throttle = self.throttle_controller_.step(vel_error, sample_time)
    brake = 0

    if linear_vel == 0.0 and current_vel < 0.1
      throttle = 0.0
      brake = 400
    elif throttle < 0.1 and vel_error < 0:
      throttle = 0.0
      decel = max(vel_error, self.decel_limit_)
      brake = abs(decel) * self.vehicle_mass_ * self.wheel_radius_
    

    return throttle, brake, steering
