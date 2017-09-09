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

# Parameters for the local searh of waypoints
max_yaw = 90 # Max relative angle we can have in forward waypoints
max_local_distance = 30.0 # Max waypoint distance we admit for local search
lookback_wps = 0
send_always = True

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
        self.final_waypoints = [] # List of final_waypoints, ready to be published
        self.ego_pose = None # Car pose
        self.msg_seq = 0 # Sequence number of /final_waypoints message

        rospy.spin()

    def _update_next_waypoint(self):
        """
        Update next_waypoint based on base_waypoints and ego_pose
        @return true if a new waypoint has been detected, false otherwise
        """
        if not self.base_waypoints or not self.ego_pose:
            return False

        prev_waypoint = self.next_waypoint

        # Get ego car variables
        ego_x = self.ego_pose.position.x
        ego_y = self.ego_pose.position.y
        ego_theta = math.atan2(self.ego_pose.orientation.y, self.ego_pose.orientation.x)

        # If we do have a next_waypoint, we start looking from it, and we stop looking
        # as soon as distance increases. Otherwise we do a full search

        wp = None
        yaw = 0
        dist = 1000000 # Long number
        if self.next_waypoint:
            idx_offset = self.next_waypoint - lookback_wps
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
            wp_theta = math.atan2(wp_y - ego_y, wp_x - ego_x)
            wp_yaw = math.degrees(wp_theta - ego_theta)

            # For debugging purposes:
            self.base_waypoints[idx].pose.header.frame_id = '{0:.2f},{1:.2f}'.format(wp_d, wp_yaw)

            if abs(wp_yaw) > max_yaw:
                continue
            if wp_d < dist:
                dist = wp_d
                wp = idx
                yaw = wp_yaw
            elif not full_search:
                # Distance is increasing. If the waypoint makes sense, just use it and break
                if dist < max_local_distance:
                    break; # We found a point
                else:
                    # We seem to have lost track. We search again
                    rospy.logwarn("Waypoint updater lost track (more than %.1f m after %d waypoints). Going back to global search.", dist, i+1)
                    full_search = True


            #rospy.loginfo("Candidate wp (%d,%.1f,%.1f)", idx, wp_d, wp_yaw)

        if wp is not None:
            self.next_waypoint = wp

        if full_search or prev_waypoint != self.next_waypoint:
            rospy.logdebug("New next wp (%d,%.1f,%.1f) after searching %d points", wp, dist, yaw, i)

        return (prev_waypoint != self.next_waypoint)
        #return True

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
        #rospy.logdebug("Received pose (%f,%f)", self.ego_pose.position.x, self.ego_pose.position.y)

        if not self.base_waypoints:
            # Nothing to do yet
            rospy.logwarn("We do not have base_waypoints yet: cannot process pose")
            return

        # 2. Find next_waypoint based on ego position & orientation
        if self._update_next_waypoint() or send_always:
            # If it has changed...
            #rospy.loginfo("Next waypont is %d", self.next_waypoint)

            # 3. Generate the list of next LOOKAHEAD_WPS waypoints
            num_base_wp = len(self.base_waypoints)
            waypoint_idx = [idx % num_base_wp for idx in range(self.next_waypoint,self.next_waypoint+LOOKAHEAD_WPS)]
            final_waypoints = [self.base_waypoints[wp] for wp in waypoint_idx]

            # 4. TODO: Update velocity for them
            # [pablo] I think this is NOT needed for "Waypoint Updater Node (Partial)". But not sure...

            # 5. Store (to allow for periodic publishing)
            self.final_waypoints = final_waypoints

            # 6. Publish waypoints to "/final_waypoints"
            self.publish_msg()


    def publish_msg(self):
            waypoint_msg = Lane()
            waypoint_msg.header.seq = self.msg_seq
            waypoint_msg.header.stamp = rospy.Time.now()
            waypoint_msg.header.frame_id = '/world'
            waypoint_msg.waypoints = self.final_waypoints
            self.final_waypoints_pub.publish(waypoint_msg)
            self.msg_seq += 1


    def waypoints_cb(self, msg):
        """
        Receive and store the whole list of waypoints.
        """
        waypoints = msg.waypoints
        hd = msg.header
        if not self.base_waypoints:
            rospy.logdebug("[WPS %d](%s,%s) Got %d waypoints", hd.seq, str(hd.stamp), hd.frame_id, len(waypoints))
            # Stamp waypoint index in PoseStamped and TwistStamped headers of internal messages
            for idx in range(len(waypoints)):
                waypoints[idx].pose.header.seq = idx
                waypoints[idx].twist.header.seq = idx
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
