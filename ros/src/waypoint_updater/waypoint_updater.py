#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

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

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        # Add Subscribers and Publisher
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # Add State variables
        self.base_waypoints = []  # List of waypoints, as received from /base_waypoints
        self.next_waypoint = None # Next waypoint in car direction
        self.ego_pose = None # Car pose

        rospy.spin()

    def _update_next_waypoint(self):
        """
        Update next_waypoint based on base_waypoints and ego_pose
        """
        if not self.base_waypoints or not self.ego_pose:
            return

        # Get ego car variables
        ego_x = self.ego_pose.position.x
        ego_y = self.ego_pose.position.y
        ego_theta = math.atan2(self.ego_pose.orientation.y, self.ego_pose.orientation.x)

        # TODO: verify that, if we already have a waypoint, it is valid

        # If we do have a next_waypoint, we use it; otherwise we find the closest
        if not self.next_waypoint:
            wp = None
            dist = 1000000 # Long number
            for idx in range(len(self.base_waypoints)):
                wp_x = self.base_waypoints[idx].pose.pose.position.x
                wp_y = self.base_waypoints[idx].pose.pose.position.y
                wp_d = math.sqrt((ego_x - wp_x)**2 + (ego_y - wp_y)**2)
                if wp_d < dist:
                    dist = wp_d
                    wp = idx
            self.next_waypoint = wp

        # Now we check the orientation, in case next_waypoint is behind us
        wp_x = self.base_waypoints[self.next_waypoint].pose.pose.position.x
        wp_y = self.base_waypoints[self.next_waypoint].pose.pose.position.y

        heading = math.atan2(wp_y - ego_y, wp_x - ego_x)
        angle = abs(ego_theta - heading)

        # It the angle between heading (the relative angle of waypoint from the car)
        # and theta (car current pose) is more than 90 deg, then the point is backwards
        # and we pick the next one. This is valid even when we started from an
        # existing next_waypoint (to find out whether ego car has gone beyond)
        if angle > math.pi/4:
            self.next_waypoint += 1

    def pose_cb(self, msg):
        """
        - Receive ego car pose
        - Update next_waypoint
        - Generate the list of the next LOOKAHEAD_WPS waypoints
        - (FUTURE) Update velocity for them
        - Publish them to "/final_waypoints"
        """

        # 1. Receive and store ego car pose
        self.ego_pose = msg.pose

        if not self.base_waypoints:
            # Nothing to do yet
            rospy.logwarn("We do not have base_waypoints yet: cannot process pose")
            return

        # 2. Find next_waypoint based on ego position & orientation
        self._update_next_waypoint()

        # 3. Generate the list of next LOOKAHEAD_WPS waypoints
        num_base_wp = len(self.base_waypoints)
        waypoint_idx = [idx % num_base_wp for idx in range(self.next_waypoint,self.next_waypoint+LOOKAHEAD_WPS)]

        final_waypoints = [self.base_waypoints[wp] for wp in waypoint_idx]

        # 4. TODO: Update velocity for them
        # [pablo] I think this is NOT needed for "Waypoint Updater Node (Partial)". But not sure...

        # 5. Publish waypoints to "/final_waypoints"
        waypoint_msg = Lane()
        waypoint_msg.waypoints = final_waypoints
        self.final_waypoints_pub.publish(waypoint_msg)


    def waypoints_cb(self, msg):
        """
        Receive and store the whole list of waypoints.
        """
        if not self.base_waypoints:
            waypoints = msg.waypoints
            #rospy.loginfo("Received a total of %d /base_waypoints. Storing them", len(waypoints))
            self.base_waypoints = waypoints
        else:
            # FIXME: [pablo] As CSV file doesn't change, I'm not reloading waypoints anymore.
            # Not sure if this is a good idea...
            pass

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
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
