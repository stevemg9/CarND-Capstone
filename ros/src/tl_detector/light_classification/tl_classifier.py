from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import os

class TLClassifier(object):
    def __init__(self, is_site,model_choice_num):

        #TODO load classifier
        graphs_dir = os.path.dirname(os.path.realpath(__file__)) + '/frozen_graphs/'
        if is_site:
            if  (model_choice_num == 1) : # ssd_inception
                model_path = graphs_dir + 'graph1b_real_ssd_inception_20k/frozen_inference_graph.pb'
            elif(model_choice_num == 2) : # faster_rcnn_inception
                model_path = graphs_dir + 'graph2b_real_faster_rcnn_inception_20k/frozen_inference_graph.pb'
            elif(model_choice_num == 3) : # faster_rcnn_resnet
                model_path = graphs_dir + 'graph3b_real_faster_rcnn_resnet_20k/frozen_inference_graph.pb'
            else : # ssd_inception is the default
                model_path = graphs_dir + 'graph1b_real_ssd_inception_20k/frozen_inference_graph.pb'
        else:
            if  (model_choice_num == 1) : # ssd_inception
                model_path = graphs_dir + 'graph1a_sim_ssd_inception_20k/frozen_inference_graph.pb'
            elif(model_choice_num == 2) : # faster_rcnn_inception
                model_path = graphs_dir + 'graph2a_sim_faster_rcnn_inception_20k/frozen_inference_graph.pb'
            elif(model_choice_num == 3) : # faster_rcnn_resnet
                model_path = graphs_dir + 'graph3a_sim_faster_rcnn_resnet_20k/frozen_inference_graph.pb'
            else : # ssd_inception is the default
                model_path = graphs_dir + 'graph1a_sim_ssd_inception_20k/frozen_inference_graph.pb'

        self.session = None
        #Deal with the model_graph
        self.model_graph = tf.Graph()
        with self.model_graph.as_default():
            with tf.gfile.GFile(model_path, 'rb') as f:
                graph_def = tf.GraphDef() #should this be outside tf.gfile.Gfile ???
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
            self.image_tensor      = self.model_graph.get_tensor_by_name('image_tensor:0')
            self.detection_boxes   = self.model_graph.get_tensor_by_name('detection_boxes:0')
            self.detection_scores  = self.model_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.model_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections    = self.model_graph.get_tensor_by_name('num_detections:0')



    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        with self.model_graph.as_default():
            with tf.Session(graph=self.model_graph) as sess:
                self.session = sess
                image_expand_dims   = np.expand_dims(image, axis=0)
                (boxes, scores, classes, num_detections) = self.session.run([self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                                                                          feed_dict={self.image_tensor: image_expand_dims})

        boxes   = np.squeeze(boxes)
        scores  = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        # Return the int of the traffic light if detected else UNKNOWN
        if classes[0]   == 1:
            return TrafficLight.GREEN
        elif classes[0] == 2:
            return TrafficLight.RED
        elif classes[0] == 3:
            return TrafficLight.YELLOW
        else:
            return TrafficLight.UNKNOWN

        return TrafficLight.UNKNOWN
