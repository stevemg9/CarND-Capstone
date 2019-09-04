import rospy
from yaw_controller import YawController
from lowpass import LowPassFilter
from pid import PID

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
  def __init__(self, vehicle_mass,
                     brake_deadband,
                     decel_limit,
                     accel_limit,
                     wheel_radius,
                     wheel_base,
                     steer_ratio,
                     max_lat_accel,
                     max_steer_angle):

    self.yaw_controller = YawController(wheel_base,
                                         steer_ratio,
                                         0.1,
                                         max_lat_accel,
                                         max_steer_angle)
    # Constants for PID controller
    kp = 0.3
    ki = 0.1
    kd = 0.0
    mn = 0.0
    mx = 0.2
    self.throttle_controller = PID(kp, ki, kd, mn, mx)

    # Constants for Low Pass Filter
    tau = 0.5
    ts  = 0.02
    self.vel_lpf = LowPassFilter(tau, ts)

    # Controller Member Variables
    self.vehicle_mass   = vehicle_mass
    self.brake_deadband = brake_deadband
    self.decel_limit    = decel_limit
    self.accel_limit    = accel_limit
    self.wheel_radius   = wheel_radius
    self.last_vel       = 0.0
    self.last_time      = rospy.get_time()

  def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):

    # Reset Controller if DBW is disengaged
    if not dbw_enabled:
      self.throttle_controller.reset()
      return 0.0, 0.0, 0.0
    
    # Get current velocity and steering values
    current_vel = self.vel_lpf.filt(current_vel)
    steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)

    # Calculate error between target and current velocity
    vel_error = linear_vel - current_vel
    self.last_vel = current_vel
    # Calculate elapsed time
    current_time = rospy.get_time()
    sample_time = current_time - self.last_time
    self.last_time = current_time

    # Get throttle value from throttle controller
    throttle = self.throttle_controller.step(vel_error, sample_time)
    brake = 0

    # Set throttle and brake values
    if linear_vel == 0.0 and current_vel < 0.1:
      throttle = 0.0
      brake = 400
    elif throttle < 0.1 and vel_error < 0:
      throttle = 0.0
      decel = max(vel_error, self.decel_limit)
      brake = abs(decel) * self.vehicle_mass * self.wheel_radius
    

    return throttle, brake, steering
