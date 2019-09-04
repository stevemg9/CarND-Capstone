#!/usr/bin/env python

#Note in order for tl_detector.py to actually get called and work. In the simulator, uncheck the  Manual mode and be sure to check the Camera mode
#The /final_waypoint topic will get published only if the camera mode has been checked in the simulator
import rospy
from scipy.spatial     import KDTree
from std_msgs.msg      import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg     import TrafficLightArray, TrafficLight
from styx_msgs.msg     import Lane
from sensor_msgs.msg   import Image
from cv_bridge         import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml

STATE_COUNT_THRESHOLD = 3
DEBUG = 0
TRAINING = True

class TLDetector(object):
  def __init__(self):
    rospy.init_node('tl_detector')
    # Creating ROS Subscribers
    sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
    sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
    sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
    sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)

    # Creating ROS Publishers
    self.upcoming_traffic_light_waypoint_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

    # Config String
    config_string = rospy.get_param("/traffic_light_config")
   
    # Class Member Variables
    self.pose             = None
    self.base_waypoints   = None
    self.waypoints_2d     = None
    self.waypoint_tree    = None
    self.lights           = []
    self.camera_image     = None
    self.config           = yaml.load(config_string)
    self.is_site          = self.config["is_site"]
    self.model_source_num = 1 #1: ssd_inception(also default), 2:faster_rcnn_inception
    self.bridge           = CvBridge()
    
    # Class members for Classifier
    self.light_classifier = TLClassifier(self.is_site, self.model_source_num)
    self.listener         = tf.TransformListener()
    self.state            = TrafficLight.UNKNOWN
    self.last_state       = TrafficLight.UNKNOWN
    self.last_wp          = -1
    self.state_count      = 0
    rospy.spin()


  def pose_cb(self, msg):
    self.pose = msg



  def waypoints_cb(self, base_waypts):
    # TODO: Implement
    # Stores the waypoints in the object.
    # So will you store the waypoints everytime the waypoints come in.
    # Ans: This is a latched subscriber. So they dont get stored everytime . This is good because the base waypoint are not changing everytime
    self.base_waypoints = base_waypts

    # Take a chunk of base_waypoints take a chunk of them. Take the first 200 of them that are in front of the car as reference
    # Figure out which waypoint is closest to the car using the KDTree. KDTree will give you the closest waypoint 'n' efficiently(its efficient. Instead of O(n) is it O(log n)
    # Lets say you have a list of 1000 waypoints and the waypoint that you want is at the very end of the list then the a direct search would take 1000 cycles.
    # But the the KDTree is of the order(log n) = log 1000= 3 cycles (but aaron brown said 10 cycles ???, maybe he was wrong)

    #use if not self.waypoints_2d because we want to initialise self.waypoints_2d much before the subsrciber is initialised (to prevent race condition)
    if not self.waypoints_2d:
      #Convert the waypoints to 2d waypoints to make it KDTree friendly
      self.waypoints_2d  = [[way_pt.pose.pose.position.x, way_pt.pose.pose.position.y] for way_pt in base_waypts.waypoints]
      self.waypoint_tree = KDTree(self.waypoints_2d)


  #Some callbacks associated with the classifier: lights information coming back
  def traffic_cb(self, msg):
    self.lights = msg.lights


  #Some callbacks associated with the classifier
  def image_cb(self, msg):
    """Identifies red lights in the incoming camera image and publishes the index
        of the waypoint closest to the red light's stop line to /traffic_waypoint

    Args:
        msg (Image): image from car-mounted camera

    """
    #Everytime we have an image coming in , what do we do about it
    #if there is an image
    self.has_image = True

    #Save the camera_image
    self.camera_image = msg

    self.traffic_light_state_routine()




  def traffic_light_state_routine(self):
    # get the waypoint-closest-to-tl and state-of-tl using self.process_traffic_lights
    light_wp, state1 = self.process_traffic_lights()

    '''
    Publish upcoming red lights at camera frequency.
    Each predicted state has to occur `STATE_COUNT_THRESHOLD` number  of times till we start using it.
    Otherwise the previous stable state is  used.
    '''
    #If you determine that the state of the traffic light has changed, start a counter
    if self.state != state1:
      self.state_count = 0
      self.state = state1
    #if the state has not been flickering and has remained the same, deal with red light(publish the NEW way point associated with this traffic light)
    #also add logic for yellow light
    elif self.state_count >= STATE_COUNT_THRESHOLD:
      self.last_state = self.state
      # If the light is red: publish the NEW way point associated with this traffic light
      # For green and yellow , we just publish -1, which means you can continue driving through
      # TODO: do something better with yellow traffic light
      if   state1 == TrafficLight.RED :
        light_wp = light_wp
        print_color = "TR: RED :"
      elif state1 == TrafficLight.GREEN :
        light_wp = -1
        print_color = "TR: GREEN :"
      elif state1 == TrafficLight.YELLOW :
        light_wp = -1
        print_color = "TR: YELLOW :"
      elif state1 == TrafficLight.UNKNOWN :
        light_wp = -1
        print_color = "TR: UNKNOWN :"
      else :
        light_wp = -1
        print_color = "TR:---------- :"
      self.last_wp = light_wp
      self.upcoming_traffic_light_waypoint_pub.publish(Int32(light_wp))
      if(DEBUG == 1):
        print(print_color,light_wp)
    # If the state has been flickering, just publish the previous way point
    else:
      self.upcoming_traffic_light_waypoint_pub.publish(Int32(self.last_wp))
      if(DEBUG == 1):
          print("Flickering State:-----",self.last_wp)
    self.state_count += 1



  def process_traffic_lights(self):
    """
    Finds closest visible traffic light, if one exists, and determines its location and color

    Returns:
        int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
        int: ID of traffic light color (specified in styx_msgs/TrafficLight)
    """
    #Closest_light is the closest traffic light
    closest_light = None
    #Each traffic light comes with the closest line: this is the stop line for the traffic light
    closest_line_wp_idx   = None

    # List of positions that correspond to the line to stop in front of for a given intersection
    stop_line_positions = self.config['stop_line_positions']
    if(self.pose):
      car_wp_idx = self.get_closest_waypoint_idx(self.pose.pose.position.x, self.pose.pose.position.y)

      #TODO find the closest visible traffic light (if one exists)
      #Iterate through all the traffic lights to find the closest one (there are only 8 traffic lights, so it is not much of a gain using the KD tree lol)
      #but the KD tree is useful when iterating through the base waypoints which are of the order of a 1000
      diff = len(self.base_waypoints.waypoints)
      for i, temp_light in enumerate(self.lights):
        # Get stop line waypoint index
        temp_line   = stop_line_positions[i]
        temp_wp_idx = self.get_closest_waypoint_idx(temp_line[0], temp_line[1])
        # Find closest stop line waypoint index
        d = temp_wp_idx - car_wp_idx
        if d >= 0 and d < diff:
          diff                = d
          closest_light       = temp_light
          closest_line_wp_idx = temp_wp_idx

    # ksw thinks: this might need to be changed
    if closest_light or TRAINING:
      closest_light_state = self.get_light_state(closest_light)
      return closest_line_wp_idx, closest_light_state

    # If the TrafficLight State is unknown just keep the car moving. The car will be tested in a very controlled environment
    return -1, TrafficLight.UNKNOWN


  #Some callbacks associated with the classifier
  def get_light_state(self, light):
    """Determines the current color of the traffic light

    Args:
        light (TrafficLight): light to classify

    Returns:
        int: ID of traffic light color (specified in styx_msgs/TrafficLight)

    """
    if(not self.has_image):
      self.prev_light_loc = None
      return False

    cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, 'bgr8')

    #Get classification
    return self.light_classifier.get_classification(cv_image)



  def get_closest_waypoint_idx(self, x,y):
    """Identifies the closest path waypoint to the given position
        https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
    Args:
        pose (Pose): position to match a waypoint to

    Returns:
        int: index of the closest waypoint in self.waypoints

    """
    #TODO implement
    #In the previos videos we set up KD trees to search for the closest waypoint in section 1(We can resuse that code)
    # Courtesy waypoints_cb
    # The waypoints_cb has already established the base_waypoints as a KDTree data structure in the self.waypoint_tree variable.
    # All we are doing is querying the waypoint_tree for the closest-base-waypoint-index based on the pose of the car (x,y)
    closest_waypt_idx = self.waypoint_tree.query([x,y] ,1)[1]

    '''
    # Check if closest is ahead or behind the vehicle
    closest_coord = self.waypoints_2d[closest_waypt_idx]
    prev_coord    = self.waypoints_2d[closest_waypt_idx -1]

    # Equation for hyperplane through closest_coords
    cl_vect   = np.array(closest_coord)
    prev_vect = np.array(prev_coord)
    pos_vect  = np.array([x,y])
    val       = np.dot(cl_vect - prev_vect, pos_vect-cl_vect)
    if val > 0: # if waypoint is behind us
      closest_waypt_idx = (closest_waypt_idx +1) % len(self.waypoints_2d)
    '''

    return closest_waypt_idx



if __name__ == '__main__':
  try:
    TLDetector()
  except rospy.ROSInterruptException:
    rospy.logerr('Could not start traffic node.')
