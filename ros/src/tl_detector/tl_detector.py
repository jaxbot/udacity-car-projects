#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose, Point
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.busy = False
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

        # Generate a list of stop line points from given configuration
        stop_line_positions = self.config['stop_line_positions']
        self.stop_line_points = [self.make_point(p[0], p[1], 0) for p in stop_line_positions]

        rospy.spin()

    def make_point(self, x, y, z):
        point = Point()
        point.x = x
        point.y = y
        point.z = z
        return point

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints.waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        if len(self.lights) < 1:
            print("Got image, ignoring because no lights are known.")
            return False
        if self.busy:
            print("Busy, skipping frame.")
            return False
        else:
            print("Processing image")
            self.busy = True
        self.has_image = True
        self.camera_image = msg
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
        self.busy = False

    def get_closest_waypoint(self, position):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        if self.waypoints is None:
            return 0
        waypoints = [waypoint.pose.pose.position for waypoint in self.waypoints]
        index, _ = self.get_closest_point(position, waypoints)
        return index

    def get_closest_point(self, needle, list_of_points):
        """Identifies the closest point in a list to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest point in list_of_points
            pose: closest point

        """
        # Brute-force solution is fine here since N is small
        minimum_distance = 9999999
        closest_point = 0
        i = 0
        for point in list_of_points:
            dist = self.distance_between_points(point, needle)
            if dist < minimum_distance:
                minimum_distance = dist
                closest_point = i
            i = i + 1
                
        return closest_point, list_of_points[closest_point]

    def get_closest_traffic_signal(self, point):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        light_poses = [light.pose.pose.position for light in self.lights]
        index, _ = self.get_closest_point(point, light_poses)
        return index

    def distance_between_points(self, point1, point2):
        """Returns the Euclidean distance between two points"""
        return math.sqrt(math.pow(point1.x - point2.x, 2) + math.pow(point1.y - point2.y, 2) + math.pow(point1.z - point2.z, 2))

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

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")

        # Classify using the TF light classifier if not available (from simulator)
        if light.state != TrafficLight.UNKNOWN and light.state != 3:
            print("Using provided state: " + str(light.state))
            return light.state
        else:
            return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose.position)
            closest_light_index = self.get_closest_traffic_signal(self.waypoints[car_position].pose.pose.position)
            light = self.lights[closest_light_index]

        if light:
            state = self.get_light_state(light)
            _, stopline = self.get_closest_point(light.pose.pose.position, self.stop_line_points)
            stopline_waypoint_index = self.get_closest_waypoint(stopline)
            return stopline_waypoint_index, state
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
