from styx_msgs.msg import TrafficLight
import cv2
import random
import time
import tensorflow as tf
import os
import numpy as np

class TLClassifier(object):
    # From label_map.pbtxt
    GREEN = 1
    YELLOW = 2
    RED = 3

    def __init__(self):
        CWD = os.path.dirname(os.path.realpath(__file__))
        PATH_TO_CKPT = CWD + '/../../../../data/exported_graphs/frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        (im_height, im_width) = image.shape[:2]
        image_np = image.reshape((im_height, im_width, 3)).astype(np.uint8)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        output_dict = self.classify_image(image_np, self.detection_graph)
        detected_boxes = []
        i = 0
        for score in output_dict["detection_scores:0"]:
            # Only add traffic light bounding boxes if we're at least 70% confident in the score.
            if score > 0.7:
                detected_boxes.append({
                        'score': score,
                        'bounding_box': output_dict["detection_boxes:0"][i],
                        'detection_class': output_dict["detection_classes:0"][i]
                    })
            i = i + 1
        
        print(detected_boxes)
        # Determine probable light color from detected bounding boxes,
        # with a basis towards yellow and red when multiple boxes are detected.
        detected_color = TrafficLight.UNKNOWN
        for box in detected_boxes:
            if box['detection_class'] == self.GREEN and detected_color == TrafficLight.UNKNOWN:
                detected_color = TrafficLight.GREEN
            if box['detection_class'] == self.YELLOW and detected_color != TrafficLight.RED:
                detected_color = TrafficLight.YELLOW
            if box['detection_class'] == self.RED:
                detected_color = TrafficLight.RED
        
        return detected_color

    def classify_image(self, image, graph):
        with graph.as_default():
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            tensor_names = ['detection_boxes:0', 'detection_classes:0', 'detection_scores:0']
            tensor_dict = {}
            for tensor_name in tensor_names:
                tensor_dict[tensor_name] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            # Run inference session on our cached TF session.
            output_dict = self.sess.run(tensor_dict,
                    feed_dict={image_tensor: np.expand_dims(image, 0)})

            # Convert float to int.
            output_dict['detection_classes:0'] = output_dict[
              'detection_classes:0'][0].astype(np.uint8)
            # Unbox tensors.
            output_dict['detection_boxes:0'] = output_dict['detection_boxes:0'][0]
            output_dict['detection_scores:0'] = output_dict['detection_scores:0'][0]
        return output_dict

