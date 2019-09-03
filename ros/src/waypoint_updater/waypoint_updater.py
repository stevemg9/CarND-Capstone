#!/usr/bin/env python
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint, TrafficLightArray
from std_msgs.msg import Int32
from scipy.spatial import KDTree
import yaml

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 5.0



class WaypointUpdater(object):
  def __init__(self):
    rospy.init_node('waypoint_updater')

    # ROS Subscribers
    rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
    rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
    rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
    #rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cheat)

    # ROS Publisher for Final Waypoints
    self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)

    # Member variables for WaypointUpdater
    self.pose = None
    self.base_waypoints  = None
    self.waypoints_2d    = None
    self.waypoint_tree   = None
    self.stopline_wp_idx = -1
    self.traffic_lights  = []
    self.stop_line_positions = []
    self.config = yaml.load(rospy.get_param("/traffic_light_config"))

    self.loop()

  def loop(self):
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
      if self.pose and self.base_waypoints:
        # Getting the closest waypoint to the vehicle
        closest_waypoint_idx = self.get_closest_waypoint_idx()
        self.publish_waypoints(closest_waypoint_idx)
      rate.sleep()

  def get_closest_waypoint_idx(self):
    # Getting current position of the vehicle
    x = self.pose.pose.position.x
    y = self.pose.pose.position.y

    # Finding closest waypoint to vehicle in KD Tree
    closest_idx = self.waypoint_tree.query([x, y], 1)[1]

    # Extracting the closest coordinate and the coordinate from the previous
    # index
    closest_coord = self.waypoints_2d[closest_idx]
    prev_coord = self.waypoints_2d[closest_idx - 1]

    # Constructing hyperplane throught the closest coordinates
    cl_vect = np.array(closest_coord)
    prev_vect = np.array(prev_coord)
    pos_vect = np.array([x, y])

    val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)

    # If waypoint is behind the vehicle, get the next waypoint
    if (val > 0):
      closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
    return closest_idx

  def publish_waypoints(self, closest_idx):
    #lane = Lane()
    #lane.waypoints = self.base_waypoints.waypoints[closest_idx : (closest_idx + LOOKAHEAD_WPS) len(self.waypoints_2d)]
    final_lane = self.generate_lane()
    self.final_waypoints_pub.publish(final_lane)

  def generate_lane(self):
    lane = Lane()

    closest_idx = self.get_closest_waypoint_idx()
    farthest_idx = closest_idx + LOOKAHEAD_WPS
    base_lane = self.base_waypoints.waypoints[closest_idx : farthest_idx]

    if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
      lane.waypoints = base_lane
    else:
      lane.waypoints = self.decelerate_waypoints(base_lane, closest_idx)
    #for light in self.traffic_lights:
    #  if light[0] > closest_idx and light[0] < farthest_idx and light[1] == 0:
    #    self.stopline_wp_idx = light[0]
    #   lane.waypoints = self.decelerate_waypoints(base_lane, closest_idx)
    #    return lane

    # lane.waypoints = base_lane
    return lane

  def decelerate_waypoints(self, waypoints, closest_idx):
    temp = []
    for i, wp in enumerate(waypoints):
      p = Waypoint()
      p.pose = wp.pose

      stop_idx = max(self.stopline_wp_idx - closest_idx - 3, 0)
      dist = self.distance(waypoints, i, stop_idx)
      vel = math.sqrt(MAX_DECEL * dist)
      if vel < 1.0:
        vel = 0

      p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
      temp.append(p)
    
    return temp


  def pose_cb(self, msg):
    # Get pose from ROS message
    self.pose = msg


  def waypoints_cb(self, waypoints):
    # Getting waypoints from ROS message
    self.base_waypoints = waypoints
    if not self.waypoints_2d:
      self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
      self.waypoint_tree = KDTree(self.waypoints_2d)

  def traffic_cb(self, msg):
    self.stopline_wp_idx = msg.data

  #def get_closest_waypoint_idx_to_tl(self, x, y):
  #  # Finding closest waypoint to vehicle in KD Tree
  #  closest_idx = self.waypoint_tree.query([x, y], 1)[1]
  #
  # # Extracting the closest coordinate and the coordinate from the previous
  #  # index
  #  closest_coord = self.waypoints_2d[closest_idx]
  #  prev_coord = self.waypoints_2d[closest_idx - 1]
  #
  #  # Constructing hyperplane throught the closest coordinates
  #  cl_vect = np.array(closest_coord)
  #  prev_vect = np.array(prev_coord)
  #  pos_vect = np.array([x, y])
  #
  #  val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)
  #
  #  # If waypoint is behind the vehicle, get the next waypoint
  #  if (val > 0):
  #    closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
  #  return closest_idx

  #def traffic_cheat(self, msg):
  #  if not self.traffic_lights:
  #    self.stop_line_positions = self.config['stop_line_positions']
  #    for light in msg.lights:
  #      min_idx = -1
  #      min_dist = 999999999999
  #      for i, pos in enumerate(self.stop_line_positions):
  #        dist = math.sqrt( pow(pos[0] - light.pose.pose.position.x, 2) +
  #                          pow(pos[1] - light.pose.pose.position.y, 2))
  #        if dist < min_dist:
  #          min_dist = dist
  #          min_idx = i
  #      idx = self.get_closest_waypoint_idx_to_tl(self.stop_line_positions[min_idx][0],
  #                                                self.stop_line_positions[min_idx][1])
  #      state = light.state
  #      self.traffic_lights.append([idx, state])
  #  else:
  #    for i, light in enumerate(msg.lights):
  #      self.traffic_lights[i][1] = light.state

  def get_waypoint_velocity(self, waypoint):
    return waypoint.twist.twist.linear.x

  def set_waypoint_velocity(self, waypoints, waypoint, velocity):
    waypoints[waypoint].twist.twist.linear.x = velocity

  def distance(self, waypoints, wp1, wp2):
    dist = 0
    dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
    for i in range(wp1, wp2+1):
      dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
      wp1 = i
    return dist


if __name__ == '__main__':
  try:
    WaypointUpdater()
  except rospy.ROSInterruptException:
    rospy.logerr('Could not start waypoint updater node.')
