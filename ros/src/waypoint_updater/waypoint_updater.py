#!/usr/bin/env python
import math
import sys

import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint, TrafficLightArray, TrafficLight

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
MAX_DECEL = 0.8
STOP_DISTANCE = 25
TESTING = False

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)

        if TESTING:
            rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray,
                             self.traffic_lights_cb, queue_size=1)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.base_waypoints = None
        self.traffic_lights = None
        self.traffic_lights_waypoints = []
        self.red_traffic_light_waypoint = -1
        self.max_velocity = self.kmph2mps(rospy.get_param('/waypoint_loader/velocity'))

        rospy.spin()

    def pose_cb(self, msg):
        current_car_wp_index = self.get_closest_waypoint(msg.pose)

        start = current_car_wp_index + 1
        stop = min(len(self.base_waypoints.waypoints),
                   start + LOOKAHEAD_WPS)
        if self.is_red_traffic_light_ahead(start, stop):
            stop_waypoint = self.get_stop_waypoint(self.red_traffic_light_waypoint,
                                                   STOP_DISTANCE)

            for i in range(start, stop_waypoint + 1):
                dist = self.distance(
                    waypoints=self.base_waypoints.waypoints,
                    wp1=i,
                    wp2=stop_waypoint)
                vel = math.sqrt(2 * MAX_DECEL * dist)
                vel = min(vel, self.get_waypoint_velocity(
                                        self.base_waypoints.waypoints[i]))

                self.set_waypoint_velocity(self.base_waypoints.waypoints, i, vel)

            for i in range(stop_waypoint, self.red_traffic_light_waypoint + 1):
                self.set_waypoint_velocity(self.base_waypoints.waypoints, i, 0)
            stop = self.red_traffic_light_waypoint
        else:
            for i in range(start, stop):
                self.set_waypoint_velocity(self.base_waypoints.waypoints, i,
                                           self.max_velocity)

        final_waypoints = Lane()
        final_waypoints.waypoints = self.base_waypoints.waypoints[start:stop]
        self.final_waypoints_pub.publish(final_waypoints)

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints

    def traffic_cb(self, msg):
        self.red_traffic_light_waypoint = msg

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def traffic_lights_cb(self, msg):
        if self.traffic_lights is None:
            for light in msg.lights:
                light_index = self.get_closest_waypoint(light.pose.pose)
                self.traffic_lights_waypoints.append(light_index)
            rospy.logerr(self.traffic_lights_waypoints)
        self.traffic_lights = msg

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

    def get_stop_waypoint(self, red_light_waypoint, stop_distance):
        for i in range(red_light_waypoint, 0, -1):
            dist = self.distance(
                    self.base_waypoints.waypoints,
                    i,
                    red_light_waypoint)

            if dist >= stop_distance:
                return i

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        min_dist = sys.maxsize
        closest_wp_index = 0

        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(len(self.base_waypoints.waypoints)):
            dist = dl(pose.position, self.base_waypoints.waypoints[i].pose.pose.position)
            if dist < min_dist:
                min_dist = dist
                closest_wp_index = i

        return closest_wp_index

    def kmph2mps(self, velocity_kmph):
        return (velocity_kmph * 1000.) / (60. * 60.)

    def is_red_traffic_light_ahead(self, wp_start, wp_stop):
        is_red_traffic_light_ahead = False
        wp_start = wp_start + STOP_DISTANCE

        if TESTING is not True:
            if (self.red_traffic_light_waypoint >= wp_start and
                self.red_traffic_light_waypoint <= wp_stop):
                is_red_traffic_light_ahead = True
                return is_red_traffic_light_ahead

        if self.traffic_lights is None:
            return is_red_traffic_light_ahead

        for i, waypoint_index in enumerate(self.traffic_lights_waypoints):
            if waypoint_index >= wp_start and waypoint_index <= wp_stop:
                if self.traffic_lights.lights[i].state == TrafficLight.RED:
                    is_red_traffic_light_ahead = True
                    self.red_traffic_light_waypoint = waypoint_index
                    break

        return is_red_traffic_light_ahead

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
