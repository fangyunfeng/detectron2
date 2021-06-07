from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import cv2
import sys
import torch
import numpy as np
import pydicom
import os.path as osp
from copy import deepcopy
import torch.nn.functional as F
sys.path.insert(0, ".")
#from ct_iterator import CTIterator
from detectron2.engine import CTIterator
from detectron2.structures import BoxMode
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN

__all__ = ["MediastinumDetector", "build_mediastinum_detector"]

gpu_id = 1
conf = {"2DStage1": {
            "device": gpu_id,
            "view": "axial",
            "nms_thresh": 0.1,
            "num_class": 15,
            "thresh": [0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            "config_file": "detectron2/config//mask_rcnn_V_57_FPN_1x_3dce.yaml"
            }}

class MePredictor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = GeneralizedRCNN(self.cfg)
        self.model.eval()
        self.model.to(torch.device(cfg.MODEL.DEVICE))
        checkpoint = torch.load(cfg.MODEL.WEIGHTS,map_location=lambda storage, loc: storage.cuda(gpu_id))# map_location=torch.device("cpu"))
        self.model.load_state_dict(checkpoint['model'], strict=False)

    def predict(self, original_image):
        with torch.no_grad():
            inputs = []
            for i in range(len(original_image)):
                image = original_image[i]
                height, width = image.shape[-2:]
                inputs.append({"image": image, "height": height, "width": width})
            predictions = self.model(inputs)
            return predictions

class MSDetector(object):
    def __init__(self, config, gpu_id, show_mask_heatmaps=False):
        self.slot = 8  # column id of det scores in result tensor
        self.device = gpu_id
        self.thresh = config['thresh']
        self.view = config['view']
        self.nms_thresh = config['nms_thresh']
        self.cfg = self.setup(config['config_file'])
        # get necessary params from model config
        self.pred = MePredictor(self.cfg)

        self.input_size = self.cfg.INPUT.MIN_SIZE_TEST
        self.batch_size = self.cfg.SOLVER.IMS_PER_BATCH
        self.input_channels = self.cfg.INPUT.SLICE_NUM

    def setup(self, config_file):
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.freeze()
        return cfg

    def inference(self, image_tensor):
        """
        run detector on all/selected slices of a CT
        :param ct: DxHxW, already on target device, dtype should be torch.uint8
        :param slice_inds: list of slices to run, torch.long
        :return: detection results, Nx7, (x, y, z, w, h, d), torch.tensor
        """
        ct_iter = CTIterator(image_tensor, self.input_size, self.device,
                        view=self.view,
                        in_channels=self.input_channels,
                        batch_size=self.batch_size)

        boxes, rois, slice_ids, scores, labels = [], [], [], [], []
        for i, batch in enumerate(ct_iter):
            inputs = []
            for i_bach in range(len(batch)):
                image = batch[i_bach]
                height, width = image.shape[-2:]
                inputs.append({"image": image, "height": height, "width": width})
            #predictions = self.model(inputs, image_size)
            #import pdb;pdb.set_trace()
            s1 = time.time()
            predictions = self.pred.predict(batch)
            torch.cuda.synchronize()
            print('pre_time:',time.time()-s1)

prop_detector = MSDetector(conf['2DStage1'], gpu_id = gpu_id)
spacing = (0.53125, 0.53125, 1.258)
image_tensors = torch.tensor(np.load('tests/vovnet/image.npz')['data'])
image_size = [512, 512]
s = time.time()
dets_ct = prop_detector.inference(image_tensors)
print('inference time: ', time.time() - s)
