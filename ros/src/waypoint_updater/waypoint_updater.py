#!/usr/bin/env python
import math
import sys
from itertools import cycle, islice

import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, TwistStamped
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
STOP_DISTANCE = 2
VELOCITY_OFFSET = 0.5

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        self.final_waypoints_pub = rospy.Publisher('final_waypoints',
                                                   Lane,
                                                   queue_size=1)
        self.final_waypoints = Lane()
        self.base_waypoints = None
        self.red_traffic_light_waypoint = -1
        self.max_velocity = self.kmph2mps(rospy.get_param('/waypoint_loader/velocity'))
        self.current_velocity = 0.

        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)

        rospy.spin()

    def pose_cb(self, msg):
        if self.base_waypoints is None:
            return

        current_car_wp_index = self.get_closest_waypoint(msg.pose)

        start = current_car_wp_index + 1
        waypoints_size = min(len(self.base_waypoints.waypoints), LOOKAHEAD_WPS)

        self.final_waypoints.waypoints = list(islice(
                                            cycle(self.base_waypoints.waypoints),
                                            start,
                                            start + waypoints_size))
        local_red_traffic_light_waypoint = self.is_red_traffic_light_ahead(
                                            self.final_waypoints.waypoints)
        if local_red_traffic_light_waypoint >= 0:
            stop_waypoint = self.get_stop_waypoint(
                                    local_red_traffic_light_waypoint,
                                    self.final_waypoints.waypoints,
                                    STOP_DISTANCE)

            self.decrease_for_stop(self.final_waypoints.waypoints,
                              stop_waypoint,
                              local_red_traffic_light_waypoint)

            self.final_waypoints.waypoints = self.final_waypoints.waypoints[:local_red_traffic_light_waypoint]
        else:
            for i in range(waypoints_size):
                self.set_waypoint_velocity(self.final_waypoints.waypoints,
                                            i,
                                            self.max_velocity)
        self.avoid_sudden_acceleration(self.final_waypoints.waypoints)

        self.final_waypoints_pub.publish(self.final_waypoints)

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints

    def traffic_cb(self, msg):
        if msg.data >= 0:
            self.set_waypoint_as_red_traffic_light(
                self.base_waypoints.waypoints[msg.data],
                is_red_traffic_light=True)
        elif self.red_traffic_light_waypoint >= 0:
            self.set_waypoint_as_red_traffic_light(
                self.base_waypoints.waypoints[self.red_traffic_light_waypoint],
                is_red_traffic_light=False)

        self.red_traffic_light_waypoint = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def velocity_cb(self, msg):
        self.current_velocity = msg.twist.linear.x

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

    def get_stop_waypoint(self, red_light_waypoint, waypoints, stop_distance):
        for i in range(red_light_waypoint, 0, -1):
            dist = self.distance(
                    waypoints,
                    i,
                    red_light_waypoint)

            if dist >= stop_distance:
                return i

        return red_light_waypoint

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

        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)
        for i in range(len(self.base_waypoints.waypoints)):
            dist = dl(pose.position, self.base_waypoints.waypoints[i].pose.pose.position)
            if dist <= min_dist:
                min_dist = dist
                closest_wp_index = i

        return closest_wp_index

    def kmph2mps(self, velocity_kmph):
        return (velocity_kmph * 1000.) / (60. * 60.)

    def set_waypoint_as_red_traffic_light(self, waypoint, is_red_traffic_light):
        # This is hackish to mark a waypoint as a red light traffic
        if is_red_traffic_light:
            waypoint.twist.twist.angular.z = 1.0
        else:
            waypoint.twist.twist.angular.z = 0.

    def is_red_traffic_light_waypoint(self, waypoint):
        return waypoint.twist.twist.angular.z > 0.

    def is_red_traffic_light_ahead(self, waypoints):
        for i in range(len(waypoints)):
            if self.is_red_traffic_light_waypoint(waypoints[i]):
                return i

        return -1

    def avoid_sudden_acceleration(self, waypoints):
        current_velocity_squared = self.current_velocity * self.current_velocity
        for i in range(len(waypoints) - 1):
            dist = self.distance(
                waypoints=waypoints,
                wp1=i,
                wp2=i+1)
            # Assume constant acceleration
            new_vel = math.sqrt(current_velocity_squared + 2 * MAX_DECEL * dist) + VELOCITY_OFFSET
            if new_vel > self.get_waypoint_velocity(waypoints[i]):
                return

            self.set_waypoint_velocity(waypoints,
                                       i,
                                       new_vel)

    def decrease_for_stop(self, waypoints, stop_waypoint, traffic_light_waypoint):
        for i in range(0, stop_waypoint):
            dist = self.distance(
                waypoints=waypoints,
                wp1=i,
                wp2=stop_waypoint)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            vel = min(vel, self.get_waypoint_velocity(waypoints[i]))

            self.set_waypoint_velocity(waypoints, i, vel)

        for i in range(stop_waypoint, traffic_light_waypoint + 1):
            self.set_waypoint_velocity(waypoints, i, 0)

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
