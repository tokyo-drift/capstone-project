#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math
import sys
import numpy as np
from mercurial.subrepo import state

STATE_COUNT_THRESHOLD = 3
USE_CLASSIFIER = False
DISPLAY_CAMERA = False
DISABLE = False

class TLDetector(object):
###########################################################################################################################
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()

        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()
###########################################################################################################################
    def pose_cb(self, msg):
        self.pose = msg.pose
###########################################################################################################################
    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints.waypoints
###########################################################################################################################
    def traffic_cb(self, msg):
        self.lights = msg.lights
###########################################################################################################################
    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera
        """

        self.has_image = True
        self.camera_image = msg
        self.camera_image.encoding = "rgb8"

        if DISABLE:
            return

        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

###########################################################################################################################
    def get_closest_waypoint(self, position_x, position_y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        index = -1
        #Return if way points are empty
        if self.waypoints is None:
            return index

        #rospy.loginfo('tl.detector.get_closest_waypoint() searching for position (%s, %s) within %s waypoints', pose.position.x, pose.position.y, len(self.waypoints))

        smallestDistance = 0
        #Brute force!: TODO: optimze
        for i in range(0, len(self.waypoints)):
            curr_wp_pos = self.waypoints[i].pose.pose.position
            distance = self.euclidianDistance(position_x, position_y, curr_wp_pos.x, curr_wp_pos.y)
            if index == -1 or distance < smallestDistance:
                index = i
                smallestDistance = distance

        #rospy.loginfo('tl.detector.get_closest_waypoint() found at: ' + str(index) + " " + str(smallestDistance))

        return index

###########################################################################################################################
    def euclidianDistance(self, x1, y1, x2, y2):
        return math.sqrt((x1 -x2)**2 + (y1 - y2)**2)

###########################################################################################################################
    def project_to_image_plane(self, point_in_world_x, point_in_world_y, point_in_world_z):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """
        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # get transform between pose of camera and world frame
        trans = None
        rot = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                  "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link",
                  "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

        if trans == None or rot == None:
            return -1,-1

        #Use tranform and rotation to calculate 2D position of light in image
        world_point = np.array([[point_in_world_x, point_in_world_y, point_in_world_z]])

        camera_mat = np.matrix([[fx, 0,  image_width/2],
                               [0, fy, image_height/2],
                               [0,  0,            1]])
        distCoeff = None

        rot_vec = self.QuaterniontoRotationMatrix(rot) # 4x1 -> quaternion to rotation matrix at z-axis
        #print("QtoR: " + str(rot_vec))

        ret, _ = cv2.projectPoints(world_point, rot_vec, np.array(trans), camera_mat, distCoeff)

        #Unpack values and return
        ret = ret.reshape(2,)
        return  int(round(ret[1])), int(round(ret[0]))

###########################################################################################################################
    def QuaterniontoRotationMatrix(self, q):
        '''Calculates the Rotation Matrix from Quaternion
        s is the real part
        x, y, z  are the complex elements'''
        #https://www.uni-koblenz.de/~cg/veranst/ws0001/sem/Bartz.pdf Chap. 1.2.6

        x, y, z, s = q

        #Precalculate repeatedly used values
        x2 = x**2
        xy = x*y
        xz = x*z

        y2 = y**2
        yz = y*z

        z2 = z**2

        sz = s*z
        sy = s*y
        sx = s*x

        #Calculate rotation matrix
        R11 = 1-2.0*(y2 + z2)
        R12 = 2.0*(xy-sz)
        R13 = 2.0*(xz+sy)

        R21 = 2.0*(xy+sz)
        R22 = 1-2.0*(x2+z2)
        R23 = 2.0*(yz-sx)

        R31 = 2.0*(xz-sy)
        R32 = 2.0*(yz+sx)
        R33 = 1-2.0*(x2+y2)

        return np.matrix([[R11, R12, R13],[R21, R22, R23],[R31, R32, R33]])

###########################################################################################################################
    def get_light_state(self, light_pos_x, light_pos_y, light_pos_z):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")


        x, y = self.project_to_image_plane(light_pos_x, light_pos_y, 0)

        #Show car image
        if DISPLAY_CAMERA:
            image_tmp = np.copy(cv_image)
            #Draw a circle
            cv2.circle(image_tmp, (x,y), 20, (255,0,0), thickness=2 )
            cv2.imshow('image', image_tmp)
            cv2.waitKey(1)

        #TODO use light location to zoom in on traffic light in image

        #Get classification
        return self.light_classifier.get_classification(cv_image)
###########################################################################################################################
    def get_nearest_traffic_light(self, waypoint_start_index):
        traffic_light = None
        traffic_light_positions = self.config['light_positions']

        last_index = sys.maxsize

        #TODO: Only one complete circle, no minimum distance considered, yet
        for i in range(0, len(traffic_light_positions)):
            index = self.get_closest_waypoint(float(traffic_light_positions[i][0]), float(traffic_light_positions[i][1]))
            if index > waypoint_start_index and index < last_index:
                last_index = index;
                traffic_light = traffic_light_positions[i]

        return traffic_light, last_index
###########################################################################################################################
    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(self.pose):
            #find the closest visible traffic light (if one exists)
            car_position = self.get_closest_waypoint(self.pose.position.x, self.pose.position.y)

            light_pos = None
            if car_position > 0:
                light_pos, light_waypoint = self.get_nearest_traffic_light(car_position)

                if light_pos:
                    #rospy.loginfo("Next traffic light ahead from waypoint " + str(car_position) +
                    #              " is at position " + str(light_pos) + " at waypoint " + str(light_waypoint))
                    state = TrafficLight.UNKNOWN
                    if USE_CLASSIFIER:
                        state = self.get_light_state(light_pos[0], light_pos[1], 0)
                    else:
                        for light in self.lights:
                            ''' If position of the light from the yaml file and one roperted via
                                /vehicle/traffic_lights differs only within 30 m consider them as same '''
                            if self.euclidianDistance(light.pose.pose.position.x, light.pose.pose.position.y, light_pos[0], light_pos[1]) < 30:
                                state = self.get_light_state(light.pose.pose.position.x, light.pose.pose.position.y, light.pose.pose.position.z)
                                state = light.state

                    return light_waypoint, state

        self.waypoints = None #TODO: Is this really needed?
        return -1, TrafficLight.UNKNOWN
###########################################################################################################################
if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
###########################################################################################################################
