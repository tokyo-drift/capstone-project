from styx_msgs.msg import TrafficLight

import tensorflow as tf
import os
import cv2
import numpy as np
import rospy

from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Bool
from sensor_msgs.msg import Image

# Function to load a graph from a protobuf file
def _load_graph(graph_file, config):
    with tf.Session(graph=tf.Graph(), config=config) as sess:
        gd = tf.GraphDef()
        with tf.gfile.Open(graph_file, 'rb') as f:
            data = f.read()
            gd.ParseFromString(data)
        tf.import_graph_def(gd, name='')
        return sess.graph

# extract traffic light box with maximal confidence
def _extractBox(boxes, scores, classes, confidence, im_width, im_height):
    # Prepare stuff
    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes).astype(np.int32)
    scores = np.squeeze(scores)

    # Get bounding box with highest confidence
    maxConf = 0
    number = -1
    for i in range(boxes.shape[0]):
        if scores[i] > confidence and classes[i] == 10:
            if scores[i] > maxConf:
                maxConf = scores[i]
                number = i

    if number != -1:
        # Create a tuple for earch box
        box = tuple(boxes[number].tolist())

        # Extract box corners
        ymin, xmin, ymax, xmax = box
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                            ymin * im_height, ymax * im_height)

        # Expand them a little bit
        left = left - 5
        if left < 0:
            left = 0
        top = top - 10
        if top < 0:
            top = 0
        bottom = bottom + 10
        if bottom > im_height:
            bottom = im_height
        right = right + 5
        if right > im_width:
            right = im_width
        box = int(left), int(right), int(top), int(bottom)
        return box

    else:
        return None

class TLClassifier(object):
    def __init__(self, model_dir = None, config = {}):

        ## Get model directory and check model files
        if model_dir is None:
            import rospkg
            rp = rospkg.RosPack()
            model_dir = os.path.join(rp.get_path('tl_detector'), 'model')
        rospy.loginfo('Using model directory {}'.format(model_dir))

        detection_model_path = os.path.join(model_dir, 'model_detection.pb')
        if not os.path.exists(detection_model_path):
            rospy.logerr('Detection model not found at {}'.format(detection_model_path))

        classification_model_path = os.path.join(model_dir, 'model_classification.pb')
        if not os.path.exists(classification_model_path):
            rospy.logerr('Classification model not found at {}'.format(classification_model_path))

        # Activate optimizations for TF
        self.config = tf.ConfigProto()
        jit_level = tf.OptimizerOptions.ON_1
        self.config.graph_options.optimizer_options.global_jit_level = jit_level

        self.graph_detection = _load_graph(detection_model_path, self.config)
        self.graph_classification = _load_graph(classification_model_path, self.config)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.graph_detection.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected
        self.detection_boxes = self.graph_detection.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects
        self.detection_scores = self.graph_detection.get_tensor_by_name('detection_scores:0')

        # This is the class from MS COCO dataset, we only need class 10 = traffic light
        self.detection_classes = self.graph_detection.get_tensor_by_name('detection_classes:0')

        # Get input and output tensors for the classification
        self.in_graph = self.graph_classification.get_tensor_by_name('input_1_1:0')
        self.out_graph = self.graph_classification.get_tensor_by_name('output_0:0')

        # Model index to TLD message
        self.index2msg = {0: TrafficLight.RED, 1: TrafficLight.GREEN, 2: TrafficLight.YELLOW}
        self.index2color = {0: (255, 0, 0), 1: (0, 255, 0), 2: (255, 255, 0)}

        ## subscriber to on/off traffic light image
        ## rostopic pub -1 /tld/publish_traffic_lights std_msgs/Bool True
        ## rosrun image_view image_view image:=/tld/traffic_light
        self.publish_traffic_light = False
        self.publish_traffic_light_sub = rospy.Subscriber('/tld/publish_traffic_lights', Bool, self.publish_traffic_light_cb)
        self.traffic_light_pub = rospy.Publisher('/tld/traffic_light', Image, queue_size = 1)
        self.bridge = CvBridge()

    def publish_traffic_light_cb(self, msg):
        self.publish_traffic_light = bool(msg)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Load image and convert

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        im_height, im_width, _ = image.shape
        image_expanded = np.expand_dims(image, axis=0)

        # Detection
        with tf.Session(graph=self.graph_detection, config=self.config) as sess:
            # Do the inference
            boxes, scores, classes = sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes],
                feed_dict={self.image_tensor: image_expanded})
            # Extract box and append to list
            box = _extractBox(boxes, scores, classes, 0.1, im_width, im_height)

        if box == None:
            return TrafficLight.UNKNOWN

        # Classification
        with tf.Session(graph=self.graph_classification, config=self.config) as sess:
            left, right, top, bottom = box
            img_crop = image[top:bottom, left:right]
            traffic_light = cv2.resize(img_crop, (32, 32))
            sfmax = list(sess.run(tf.nn.softmax(self.out_graph.eval(feed_dict={self.in_graph: [traffic_light]}))))
            sf_ind = sfmax.index(max(sfmax))

            ## add a colored bbox and publish traffic light if needed
            if self.publish_traffic_light:
                cv2.rectangle(traffic_light, (0, 0), (31, 31), self.index2color[sf_ind], 1)
                self.traffic_light_pub.publish(self.bridge.cv2_to_imgmsg(traffic_light, "rgb8"))

            return self.index2msg[sf_ind]

        return TrafficLight.UNKNOWN


if __name__ == "__main__":
    classifier = TLClassifier() # config = {'device_count': {'GPU': 0}}, log_device_placement = True
    classifier.publish_traffic_light = False

    import rospkg
    rp = rospkg.RosPack()
    images_dir = os.path.join(rp.get_path('tl_detector'), 'images')

    ## local tests
    for name in ['1505373294_515779972.png', '1505413792_582132101.png', '1505373315_368932962.png', '1505373321_652045011.png']:
        image = cv2.imread(os.path.join(images_dir, name))
        color = classifier.get_classification(image)
        print (color)
