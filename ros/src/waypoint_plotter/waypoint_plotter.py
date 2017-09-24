#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
import matplotlib.pyplot as plt
import numpy as np

class WaypointPlotter(object):
    def __init__(self):
        rospy.init_node('waypoint_plotter')


        self.current_pose = None
        self.red_light_waypoint = None
        self.final_waypoints = None
        self.base_waypoints = None

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size = 1)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb, queue_size = 1)
        rospy.Subscriber('/final_waypoints', Lane, self.final_waypoints_cb, queue_size = 1)
        rospy.Subscriber('/base_waypoints', Lane, self.base_waypoints_cb, queue_size = 1)

        fig = plt.figure(figsize=(5.6, 8))

        ax_pos = fig.add_subplot(211)
        final_waypoints, = ax_pos.plot([], [])
        base_waypoints, = ax_pos.plot([], [])
        current_position, = ax_pos.plot([], [])
        traffic_light, = ax_pos.plot([], [])

        ax_vel = fig.add_subplot(212)
        velocity, = ax_vel.plot([], [])

        fig.set_visible(True)
        fig.show()

        # Launch periodic publishing into /final_waypoints
        while not rospy.is_shutdown():

            base_waypoints.set_xdata([])
            base_waypoints.set_ydata([])
            final_waypoints.set_xdata([])
            final_waypoints.set_ydata([])
            current_position.set_xdata([])
            current_position.set_ydata([])
            traffic_light.set_xdata([])
            traffic_light.set_ydata([])

            if self.final_waypoints is not None:
                final_waypoints.set_xdata([w.pose.pose.position.x for w in self.final_waypoints])
                final_waypoints.set_ydata([w.pose.pose.position.y for w in self.final_waypoints])
                final_waypoints.set_marker('o')

            if self.current_pose is not None:
                current_position.set_xdata([self.current_pose.position.x,])
                current_position.set_ydata([self.current_pose.position.y,])
                current_position.set_marker('o')
                current_position.set_color('g')
                current_position.set_markersize(15)

            if self.red_light_waypoint is not None and self.base_waypoints is not None:
                traffic_pose = self.base_waypoints[self.red_light_waypoint].pose.pose
                traffic_light.set_xdata([traffic_pose.position.x,])
                traffic_light.set_ydata([traffic_pose.position.y,])
                traffic_light.set_marker('o')
                traffic_light.set_markersize(10)
                traffic_light.set_color('r')

            ax_pos.relim()
            ax_pos.set_aspect('equal', 'datalim')
            ax_pos.autoscale_view(True,True,True)

            if self.base_waypoints is not None:
                base_waypoints.set_xdata([w.pose.pose.position.x for w in self.base_waypoints])
                base_waypoints.set_ydata([w.pose.pose.position.y for w in self.base_waypoints])
                base_waypoints.set_linewidth(0.5)


            if self.final_waypoints is not None:
                velocity.set_xdata([i for i, w in enumerate(self.final_waypoints)])
                velocity.set_ydata([w.twist.twist.linear.x for w in self.final_waypoints])

            ax_vel.relim()
            ax_vel.autoscale_view(True,True,True)

            fig.canvas.draw_idle()
            plt.pause(0.1)


    def pose_cb(self, msg):
        self.current_pose = msg.pose

    def traffic_cb(self, msg):
         self.red_light_waypoint = msg.data if msg.data >= 0 else None

    def final_waypoints_cb(self, msg):
        self.final_waypoints = msg.waypoints

    def base_waypoints_cb(self, msg):
        self.base_waypoints = msg.waypoints


    # def _update_next_waypoint(self):
    #     """
    #     Update next_waypoint based on base_waypoints and current_pose.
    #     @return True if a valid waypoint has been updated, False otherwise
    #     """
    #     if not self.base_waypoints:
    #         rospy.logwarn("Waypoints not updated: base_waypoints not available yet.")
    #         return False

    #     if not self.current_pose:
    #         rospy.logwarn("Waypoints not updated: current_pose not available yet.")
    #         return False

    #     # Get ego car variables
    #     ego_x = self.current_pose.position.x
    #     ego_y = self.current_pose.position.y
    #     ego_theta = math.atan2(self.current_pose.orientation.y, self.current_pose.orientation.x)

    #     # If we do have a next_waypoint, we start looking from it, and we stop looking
    #     # as soon as we get a local minimum. Otherwise we do a full search across the whole track
    #     t = time.time()
    #     wp = None
    #     yaw = 0
    #     dist = 1000000 # Long number
    #     if self.next_waypoint:
    #         idx_offset = self.next_waypoint
    #         full_search = False
    #     else:
    #         idx_offset = 0
    #         full_search = True
    #     num_base_wp = len(self.base_waypoints)

    #     for i in range(num_base_wp):
    #         idx = (i + idx_offset)%(num_base_wp)
    #         wp_x = self.base_waypoints[idx].pose.pose.position.x
    #         wp_y = self.base_waypoints[idx].pose.pose.position.y
    #         wp_d = math.sqrt((ego_x - wp_x)**2 + (ego_y - wp_y)**2)

    #         if wp_d < dist:
    #             dist = wp_d
    #             wp = idx
    #             if debugging:
    #                 # Angle betwee car heading and waypoint heading
    #                 yaw = math.atan2(wp_y - ego_y, wp_x - ego_x) - ego_theta
    #         elif not full_search:
    #             # Local minimum. If the waypoint makes sense, just use it and break
    #             if dist < max_local_distance:
    #                 break; # We found a point
    #             else:
    #                 # We seem to have lost track. We search again
    #                 rospy.logwarn("Waypoint updater lost track (local min at %.1f m after %d waypoints). Going back to full search.", dist, i+1)
    #                 full_search = True

    #     if debugging:
    #         rospy.loginfo("New next wp [%d] -> (%.1f,%.1f) after searching %d points in %fs", wp, dist * math.cos(yaw), dist * math.sin(yaw), i, time.time()-t)

    #     if wp is None:
    #         rospy.logwarn("Waypoint updater did not find a valid waypoint")
    #         return False

    #     self.next_waypoint = wp
    #     return True

    # def update_and_publish(self):
    #     """
    #     - Update next_waypoint based on current_pose and base_waypoints
    #     - Generate the list of the next LOOKAHEAD_WPS waypoints
    #     - Update velocity for them
    #     - Publish them to "/final_waypoints"
    #     """
    #     # 1. Find next_waypoint based on ego position & orientation
    #     if self._update_next_waypoint():

    #         # 2. Generate the list of next LOOKAHEAD_WPS waypoints
    #         num_base_wp = len(self.base_waypoints)
    #         waypoint_idx = [idx % num_base_wp for idx in range(self.next_waypoint,self.next_waypoint+LOOKAHEAD_WPS)]
    #         final_waypoints = [self.base_waypoints[wp] for wp in waypoint_idx]

    #         # 3. If there is a red light ahead, update velocity for them
    #         if stop_on_red:
    #             # Start from original velocities
    #             self.restore_velocities(waypoint_idx)
    #             try:
    #                 red_idx = waypoint_idx.index(self.red_light_waypoint)
    #                 self.decelerate(final_waypoints, red_idx)
    #             except ValueError:
    #                 # No red light available: self.red_light_waypoint is None or not in final_waypoints
    #                 red_idx = None
    #             if debugging:
    #                 v = self.get_waypoint_velocity(final_waypoints, 0)
    #                 rospy.loginfo("Target velocity: %.1f, RL:%s wps ahead", v, str(red_idx))

    #         # 4. Publish waypoints to "/final_waypoints"
    #         self.publish_msg(final_waypoints)


    # def publish_msg(self, final_waypoints):
    #         waypoint_msg = Lane()
    #         waypoint_msg.header.seq = self.msg_seq
    #         waypoint_msg.header.stamp = rospy.Time.now()
    #         waypoint_msg.header.frame_id = '/world'
    #         waypoint_msg.waypoints = final_waypoints
    #         self.final_waypoints_pub.publish(waypoint_msg)
    #         self.msg_seq += 1

    # def pose_cb(self, msg):
    #     """
    #     Receive and store ego pose
    #     """
    #     self.current_pose = msg.pose

    # def waypoints_cb(self, msg):
    #     """
    #     Receive and store the whole list of waypoints.
    #     """
    #     t = time.time()
    #     waypoints = msg.waypoints
    #     num_wp = len(waypoints)

    #     if self.base_waypoints and self.next_waypoint is not None:
    #         # Normally we assume that waypoint list doesn't change (or, at least, not
    #         # in the position where the car is located). If that happens, just handle it.
    #         if not self.is_same_waypoint(self.base_waypoints[self.next_waypoint],
    #                                      waypoints[self.next_waypoint]):
    #             self.next_waypoint = None # We can't assume previous knowledge of waypoint
    #             self.base_waypoints = None # Just for debugging. Will be updated later
    #             rospy.logwarn("Base waypoint list changed")
    #     else:
    #         # No change. We could probably return here.
    #         pass

    #     """
    #     # -- Uncomment for debugging
    #     # Stamp waypoint index in PoseStamped and TwistStamped headers of internal messages
    #     for idx in range(len(waypoints)):
    #         waypoints[idx].pose.header.seq = idx
    #         waypoints[idx].twist.header.seq = idx
    #     """

    #     self.base_wp_orig_v = [self.get_waypoint_velocity(waypoints, idx) for idx in range(num_wp)]

    #     if debugging and not self.base_waypoints:
    #         dist = self.distance(waypoints, 0, num_wp-1)
    #         rospy.loginfo("Received: %d waypoints, %.1f m, %.1f m/wp in t=%f", num_wp, dist, dist/num_wp, time.time()-t)

    #     self.base_waypoints = waypoints

    # def traffic_cb(self, msg):
    #     """
    #     Receive and store the waypoint index for the next red traffic light.
    #     If the index is <0, then there is no red traffic light ahead
    #     """
    #     prev_red_light_waypoint = self.red_light_waypoint
    #     self.red_light_waypoint = msg.data if msg.data >= 0 else None
    #     if prev_red_light_waypoint != self.red_light_waypoint:
    #         if debugging:
    #             rospy.loginfo("TrafficLight changed: %s", str(self.red_light_waypoint))
    #         if publish_on_light_change:
    #             self.update_and_publish() # Refresh if next traffic light has changed

    # def obstacle_cb(self, msg):
    #     # Obstacle handling is not needed in this version of the project. Nothing to do here.
    #     pass

    # def restore_velocities(self, indexes):
    #     """
    #     Restore original velocities of points
    #     """
    #     for idx in indexes:
    #         self.set_waypoint_velocity(self.base_waypoints, idx, self.base_wp_orig_v[idx])

    # def decelerate(self, waypoints, stop_index):
    #     """
    #     Decelerate a list of wayponts so that they stop on stop_index
    #     """
    #     if stop_index <= 0:
    #         return
    #     dist = self.distance(waypoints, 0, stop_index)
    #     step = dist / stop_index
    #     # Generate waypoint velocity by traversing the waypoint list backwards:
    #     #  - Everything beyond stop_index will have velocity = 0
    #     #  - Before that, constant (de)cceleration is applied until reaching
    #     #    previous waypoint velocity.
    #     # We assume constant distance between consecutive waypoints for simplicity
    #     v = 0.
    #     d = 0.
    #     for idx in reversed(range(len(waypoints))):
    #         if idx < stop_index:
    #             d += step
    #             if d > STOP_DISTANCE:
    #                 v = math.sqrt(2*abs(self.accel)*(d-STOP_DISTANCE))
    #         if v < self.get_waypoint_velocity(waypoints, idx):
    #             self.set_waypoint_velocity(waypoints, idx, v)


    # def get_waypoint_velocity(self, waypoints, waypoint):
    #     return waypoints[waypoint].twist.twist.linear.x

    # def set_waypoint_velocity(self, waypoints, waypoint, velocity):
    #     waypoints[waypoint].twist.twist.linear.x = velocity

    # def distance(self, waypoints, wp1, wp2):
    #     dist = 0
    #     dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
    #     for i in range(wp1, wp2+1):
    #         dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
    #         wp1 = i
    #     return dist

    # def is_same_waypoint(self, wp1, wp2, max_d=0.5, max_v=0.5):
    #     """
    #     Compare two waypoints to see whether they are the same
    #     (within 0.5 m and 0.5 m/s)
    #     """
    #     dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
    #     ddif = dl(wp1.pose.pose.position, wp2.pose.pose.position)
    #     if ddif < max_d:
    #        return True
    #     return False

if __name__ == '__main__':
    try:
        WaypointPlotter()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
