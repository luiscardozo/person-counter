
class Model():

    def __init__(self):
        self._models = [
            {
                'name': 'person-detection-retail-0013',
                'path': f"./models/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml",
                'person-class': 1.0,
                'origin': 'intel',
                'enabled': True
            },
            {
                'name': 'ssd_mobilenet_v2_coco',
                'path': './tmp/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml',
                'person-class': 1.0,
                'origin': 'tensorflow',
                'comments': 'Lot of errors, do not detect the 2nd person',
                'enabled': True
            },
            {
                'name': 'faster_rcnn_resnet50_coco_2018_01_28',
                'path': './tmp/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.xml',
                'person-class': 1.0,
                'origin': 'tensorflow',
                'comments': 'Somewhat slow',
                'enabled': True
            },
            {
                'name': 'faster_rcnn_resnet101_ava_v2.1',
                'path': 'tmp/faster_rcnn_resnet101_ava_v2.1_2018_04_30/frozen_inference_graph.xml',
                'person-class': 12.0,
                'origin': 'tensorflow',
                'comments': 'slow, sometimes does not detect #Classes: human actions. https://research.google.com/ava/download/ava_action_list_v2.2_for_activitynet_2019.pbtxt',
                'enabled': False
            },
            {
                'name': 'faster_rcnn_resnet101_kitti_2018_01_28',
                'path': 'tmp/faster_rcnn_resnet101_kitti_2018_01_28/frozen_inference_graph.xml',
                'person-class': 2.0,
                'origin': 'tensorflow',
                'comments': 'slow, sometimes does not detect # Classes: 2 (Pedestrian). https://github.com/utiasSTARS/pykitti/blob/3661c441026f84519ded0bbfd7db5592d6e20b41/pykitti/tracking.py#L223',
                'enabled': True
            },
            {
                'name': 'ssd_inception_v2_coco',
                'path': 'tmp/ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.xml',
                'person-class': 1.0,
                'origin': 'tensorflow',
                'comments': 'sometimes does not detect',
                'enabled': True
            },
            {
                'name': '',
                'path': 'tmp/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.xml',
                'person-class': 1.0,
                'origin': 'tensorflow',
                'comments': 'Somewhat slow',
                'enabled': True
            },
            {
                'name': 'caffe-vggnet',
                'path': 'tmp/caffe/vggnet/VGG_VOC0712Plus_SSD_300x300_iter_240000.xml',
                'person-class': 15.0,
                'origin': 'caffe',
                'comments': 'slow',
                'enabled': True,
                'url': 'https://drive.google.com/file/d/0BzKzrI_SkD1_WnR2T1BGVWlCZHM/view',
            },
            {
                'name': 'faster_rcnn_resnet101_coco_2018_01_28_fp32',
                'path': 'tmp/faster_rcnn_resnet101_coco_2018_01_28_fp32/frozen_inference_graph.xml',
                'person-class': 1.0,
                'origin': 'tensorflow',
                'comments': 'slow',
                'enabled': True
            },
            {
                'name': 'faster_rcnn_resnet101_coco_2018_01_28',
                'path': 'tmp/faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.xml',
                'person-class': 1.0,
                'origin': 'tensorflow',
                'comments': 'slow',
                'enabled': True
            },
        ]

    def __iter__(self):
        self._iter = iter(self._models)
        return self

    def __next__(self):
        return next(self._iter)

    def __len__(self):
        return len(self._models)

    def get_model(self, name):
        model = None
        for m in self._models:
            if m['name'] == name:
                model = m

        return model

    def get_model_path(self, name):
        model = self.get_model(name)
        return model['path'] if model else None

    def get_model_person_class(self, name):
        model = self.get_model(name)
        return model['person-class'] if model else None

    def get_default(self):
        return self.get_model('person-detection-retail-0013')

    def get_default_model_path(self):
        return self.get_default()['path']

#DEFAULT_MODEL=f"./models/intel/person-detection-retail-0013/FP{DEFAULT_PREC}/person-detection-retail-0013.xml"
#DEFAULT_MODEL="./tmp/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml" # Lot of errors, do not detect the 2nd person
#DEFAULT_MODEL="./tmp/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.xml" # Somewhat slow
#DEFAULT_MODEL="./tmp/faster_rcnn_resnet101_ava_v2.1_2018_04_30/frozen_inference_graph.xml" # slow, sometimes does not detect #Classes: human actions. https://research.google.com/ava/download/ava_action_list_v2.2_for_activitynet_2019.pbtxt
#DEFAULT_MODEL="./tmp/onnx-mobilenetv2-1.0/mobilenetv2-1.0.xml" # only for classification
#DEFAULT_MODEL="tmp/faster_rcnn_resnet101_kitti_2018_01_28/frozen_inference_graph.xml" # slow, sometimes does not detect # Classes: 2 (Pedestrian). https://github.com/utiasSTARS/pykitti/blob/3661c441026f84519ded0bbfd7db5592d6e20b41/pykitti/tracking.py#L223
#DEFAULT_MODEL="tmp/ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.xml" # sometimes does not detect

#DEFAULT_MODEL="tmp/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.xml" # Somewhat slow

#https://github.com/zlingkang/mobilenet_ssd_pedestrian_detection
#DEFAULT_MODEL="tmp/mobilenet_ssd_pedestrian_detection/MobileNetSSD_deploy10695.xml" # nothing detected

#https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/ssd
#DEFAULT_MODEL="tmp/onnx/ssd-10.xml" #unsupported layers 'NonMaxSuppression' VPU

#https://github.com/weiliu89/caffe/tree/ssd
#https://drive.google.com/file/d/0BzKzrI_SkD1_WnR2T1BGVWlCZHM/view
#DEFAULT_MODEL="tmp/caffe/vggnet/VGG_VOC0712Plus_SSD_300x300_iter_240000.xml" # slow; label = 15.0

#DEFAULT_MODEL="tmp/faster_rcnn_resnet101_coco_2018_01_28_fp32/frozen_inference_graph.xml" # slow
#DEFAULT_MODEL="tmp/faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.xml" # slow