#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

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


"""
Global constants to control the node behaviour.

We assume:
  - Distance between consecutive waypoints ~ 0.5 to 1m
  - Max speed ~ 10 MPH = 4.47 m/s

Under this conditions, PUBLISH_RATE is set to 1 Hz to avoid impact on performance.
It won't progress more than ~5 waypoints per cycle. Waypoint follower takes care
of post-processing this message, filter-out backwards waypoints and provide a
clean twist value to the DBW node.
"""
LOOKAHEAD_WPS = 200       # Number of waypoints we will publish
PUBLISH_RATE = 1          # Publishing rate (Hz)
MAX_LOCAL_DISTANCE = 20.0 # Max waypoint distance we admit for a local minimum (m)
PUBLISH_ON_LIGHT_CHANGE = True # Force publishing if next traffic light changes

stop_on_red = True # Enable/disable stopping on red lights
debugging = True   # Set to False for release (not too verbose, but it saves some computation power)

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        # Add Subscribers and Publisher
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        # rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # Add State variables
        self.base_waypoints = []  # List of waypoints, as received from /base_waypoints
        self.next_waypoint = None # Next waypoint in car direction
        self.current_pose = None # Car pose
        self.red_light_waypoint = None # Waypoint index of the next red light
        self.msg_seq = 0 # Sequence number of /final_waypoints message

        # Launch periodic publishing into /final_waypoints
        rate = rospy.Rate(PUBLISH_RATE)
        while not rospy.is_shutdown():
            self.update_and_publish()
            rate.sleep()

    def _update_next_waypoint(self):
        """
        Update next_waypoint based on base_waypoints and current_pose.
        @return True if a valid waypoint has been updated, False otherwise
        """
        if not self.base_waypoints:
            rospy.logwarn("Waypoints not updated: base_waypoints not available yet.")
            return False

        if not self.current_pose:
            rospy.logwarn("Waypoints not updated: current_pose not available yet.")
            return False

        # Get ego car variables
        ego_x = self.current_pose.position.x
        ego_y = self.current_pose.position.y
        ego_theta = math.atan2(self.current_pose.orientation.y, self.current_pose.orientation.x)

        # If we do have a next_waypoint, we start looking from it, and we stop looking
        # as soon as we get a local minimum. Otherwise we do a full search across the whole track

        wp = None
        yaw = 0
        dist = 1000000 # Long number
        if self.next_waypoint:
            idx_offset = self.next_waypoint
            full_search = False
        else:
            idx_offset = 0
            full_search = True
        num_base_wp = len(self.base_waypoints)

        for i in range(num_base_wp):
            idx = (i + idx_offset)%(num_base_wp)
            wp_x = self.base_waypoints[idx].pose.pose.position.x
            wp_y = self.base_waypoints[idx].pose.pose.position.y
            wp_d = math.sqrt((ego_x - wp_x)**2 + (ego_y - wp_y)**2)

            if wp_d < dist:
                dist = wp_d
                wp = idx
                if debugging:
                    # Angle betwee car heading and waypoint heading
                    yaw = math.atan2(wp_y - ego_y, wp_x - ego_x) - ego_theta
            elif not full_search:
                # Local minimum. If the waypoint makes sense, just use it and break
                if dist < MAX_LOCAL_DISTANCE:
                    break; # We found a point
                else:
                    # We seem to have lost track. We search again
                    rospy.logwarn("Waypoint updater lost track (local min at %.1f m after %d waypoints). Going back to full search.", dist, i+1)
                    full_search = True

        if debugging:
            rospy.loginfo("New next wp [%d] -> (%.1f,%.1f) after searching %d points", wp, dist * math.cos(yaw), dist * math.sin(yaw), i)

        if wp is None:
            rospy.logwarn("Waypoint updater did not find a valid waypoint")
            return False

        self.next_waypoint = wp
        return True

    def decelerate(self, waypoints, stop_index):
        """
        Decelerate a list of wayponts so that they stop on stop_index
        """
        if stop_index <= 0:
            rospy.logwarn("Not enough space for a safe stop.")
            return
        dist = self.distance(wayponts, 0, stop_index)
        start_v = self.get_waypoint_velocity(wayponts[0])
        end_v = 0
        step = dist / stop_index
        if logging:
            rospy.loginfo("Decelerate %.2f->%.2f in %.1f m (%d wps at %.1f m/wp)",
                          start_v, end_v, dist, stop_index, step)
        # TODO: - Generate a smooth velocity trayectory from 0 to stop_index
        #           - Acceleration should not exceed 10 m/s^2
        #           - Jerk should not exceed 10 m/s^3
        #       - Set waypoint velocity to 0.0 from stop_index onwards
        #       - If it is not possible to stop safely, just keep going

    def update_and_publish(self):
        """
        - Update next_waypoint based on current_pose and base_waypoints
        - Generate the list of the next LOOKAHEAD_WPS waypoints
        - Update velocity for them
        - Publish them to "/final_waypoints"
        """
        # 1. Find next_waypoint based on ego position & orientation
        if self._update_next_waypoint():

            # 2. Generate the list of next LOOKAHEAD_WPS waypoints
            num_base_wp = len(self.base_waypoints)
            waypoint_idx = [idx % num_base_wp for idx in range(self.next_waypoint,self.next_waypoint+LOOKAHEAD_WPS)]
            final_waypoints = [self.base_waypoints[wp] for wp in waypoint_idx]

            # 3. If there is a red light ahead, update velocity for them
            if stop_on_red:
                try:
                    red_idx = waypoint_idx.index(self.red_light_waypoint)
                    self.decelerate(waypoints, red_idx)
                except ValueError:
                    # No red light available: self.red_light_waypoint is None or not in final_waypoints
                    pass

            # 4. Publish waypoints to "/final_waypoints"
            self.publish_msg(final_waypoints)


    def publish_msg(self, final_waypoints):
            waypoint_msg = Lane()
            waypoint_msg.header.seq = self.msg_seq
            waypoint_msg.header.stamp = rospy.Time.now()
            waypoint_msg.header.frame_id = '/world'
            waypoint_msg.waypoints = final_waypoints
            self.final_waypoints_pub.publish(waypoint_msg)
            self.msg_seq += 1

    def pose_cb(self, msg):
        """
        Receive and store ego pose
        """
        self.current_pose = msg.pose

    def waypoints_cb(self, msg):
        """
        Receive and store the whole list of waypoints.
        """
        if self.base_waypoints:
            # FIXME: [pablo] As CSV file doesn't change, I'm not reloading waypoints anymore.
            # Not sure if this is a good idea...
            return

        waypoints = msg.waypoints
        num_wp = len(waypoints)
        # Stamp waypoint index in PoseStamped and TwistStamped headers of internal messages
        for idx in range(len(waypoints)):
            waypoints[idx].pose.header.seq = idx
            waypoints[idx].twist.header.seq = idx
        self.base_waypoints = waypoints

        if debugging:
            dist = self.distance(waypoints, 0, num_wp-1)
            rospy.loginfo("Received: %d waypoints, %.1f m, %.1f m/wp", num_wp, dist, dist/num_wp)

    def traffic_cb(self, msg):
        """
        Receive and store the waypoint index for the next red traffic light.
        If the index is <0, then there is no red traffic light ahead
        """
        prev_red_light_waypoint = self.red_light_waypoint
        self.red_light_waypoint = msg.data if msg.data >= 0 else None
        # rospy.loginfo("TrafficLight received: %d -> %s", msg.data, str(self.red_light_waypoint))
        if PUBLISH_ON_LIGHT_CHANGE and prev_red_light_waypoint != self.red_light_waypoint:
            self.update_and_publish() # Refresh if next traffic light has changed

    def obstacle_cb(self, msg):
        # Obstacle handling is not needed in this version of the project. Nothing to do here.
        pass

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
