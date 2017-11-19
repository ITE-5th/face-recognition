import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cv2 import cv2

# from CMS.RPN.RPN import RPN
from CMS.faster_rcnn import network
from CMS.faster_rcnn.faster_rcnn import FasterRCNN, RPN
from L2Normalization import L2Norm
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from fast_rcnn.nms_wrapper import nms
from network import FC
from roi_pooling.modules.roi_pool import RoIPool
from rpn_msr.proposal_target_layer import proposal_target_layer as proposal_target_layer_py
from utils.blob import im_list_to_blob


def nms_detections(pred_boxes, scores, nms_thresh, inds=None):
    dets = np.hstack((pred_boxes,
                      scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, nms_thresh)
    if inds is None:
        return pred_boxes[keep], scores[keep]
    return pred_boxes[keep], scores[keep], inds[keep]


class CMSRCNN(nn.Module):
    PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
    SCALES = (600,)
    MAX_SIZE = 1000

    def __init__(self):
        super(CMSRCNN, self).__init__()
        # VGG-16 structure: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

        # Load VGG_Face into vgg_face object
        # vgg_face.load_state_dict(torch.load('models/vgg_face.pth'))

        # remove classifier layers
        # self.vgg = nn.Sequential(*(vgg_face._modules['{}'.format(i)] for i in range(31)))

        # define RPN layer
        # self.rpn = RPN(vgg=self.vgg)

        # self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
        self.pool = nn.MaxPool2d(2)
        # self.pool1 = nn.MaxPool2d(2)
        # self.pool2 = nn.MaxPool2d(2)
        # self.pool3 = nn.MaxPool2d(4)
        # self.pool4 = nn.MaxPool2d(2)
        # self.pool5 = nn.MaxPool2d(2)

        self.conv_a1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_b1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv_a2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_b2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv_a3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_b3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_c3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.seq1 = nn.Sequential(
            self.conv_a1,
            # nn.ReLU(),
            self.conv_b1,
            # nn.ReLU(),
            nn.MaxPool2d(2),

            self.conv_a2,
            # nn.ReLU(),
            self.conv_b2,
            nn.ReLU(),
            # self.pool,
            nn.MaxPool2d(2),

            self.conv_a3,
            # nn.ReLU(),
            self.conv_b3,
            # nn.ReLU(),
            self.conv_c3,
            # nn.ReLU()
        )

        self.conv_a4 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_b4 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_c4 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.seq2 = nn.Sequential(
            self.conv_a4,
            # nn.ReLU(),
            self.conv_b4,
            # nn.ReLU(),
            self.conv_c4,
            # nn.ReLU()
        )

        self.conv_a5 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_b5 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_c5 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.seq3 = nn.Sequential(
            self.conv_a5,
            # nn.ReLU(),
            self.conv_b5,
            # nn.ReLU(),
            self.conv_c5,
            # nn.ReLU()
        )

        self.rpn = RPN()

        c = 512
        self.l2norm_rpn_1 = L2Norm([66.84] * 256)
        self.l2norm_rpn_2 = L2Norm([94.52] * c)
        self.l2norm_rpn_3 = L2Norm([94.52] * c)

        self.l2norm_faces_1 = L2Norm([57.75] * c)
        self.l2norm_faces_2 = L2Norm([81.67] * c)
        self.l2norm_faces_3 = L2Norm([81.67] * c)

        self.l2norm_bodies_1 = L2Norm([57.75] * c)
        self.l2norm_bodies_2 = L2Norm([81.67] * c)
        self.l2norm_bodies_3 = L2Norm([81.67] * c)

        # TODO: remove or keep padding??
        self.conv1x1_rpn = nn.Conv2d(1280, 512, kernel_size=(1, 1))
        self.conv1x1_faces = nn.Conv2d(512 * 3, 512 * 7 * 7, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1))
        self.conv1x1_bodies = nn.Conv2d(512 * 3, 512 * 7 * 7, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1))

        # TODO: correct parameters
        self.roi_pool1 = RoIPool(7, 7, 1.0 / 16)
        self.roi_pool2 = RoIPool(7, 7, 1.0 / 16)
        self.roi_pool3 = RoIPool(7, 7, 1.0 / 16)

        self.fc1 = FC(512 * 7 * 7, 4096)
        self.fc2 = FC(4096, 4096)
        self.score_fc = FC(4096, 2, relu=False)
        self.bbox_fc = FC(4096, 4, relu=False)

        self.cross_entropy = None
        self.loss_box = None

    def forward(self, im_data, im_info, gt_boxes=None, gt_ishard=None, dontcare_areas=None):
        im_data = network.np_to_variable(im_data, is_cuda=True)
        im_data = im_data.permute(0, 3, 1, 2)

        x1 = self.seq1(im_data)
        x1_rpn = self.pool(x1)

        x2 = self.seq2(x1_rpn)
        x2_rpn = self.pool(x2)

        x3 = self.seq3(x2_rpn)

        # region RPN

        # 2x pooling
        x1_rpn = self.l2norm_rpn_1(self.pool(x1_rpn))
        # 1x pooling
        x2_rpn = self.l2norm_rpn_2(x2_rpn)
        # 0x pooling
        x3_rpn = self.l2norm_rpn_3(x3)
        # x1_rpn = self.pool4(self.pool3(x1))
        # x1_rpn = self.pool3(x1)
        # x2_rpn = self.pool5(x1)

        # x1_rpn = self.pool(x1_rpn)
        # x3_rpn = x3

        temp = torch.cat((x1_rpn, x2_rpn, x3_rpn), 1)
        # temp = torch.cat((temp, x3_rpn), 1)
        rpn_input = self.conv1x1_rpn(temp)
        rois = self.rpn(rpn_input, im_info, gt_boxes, gt_ishard, dontcare_areas)

        if self.training:
            roi_data = self.proposal_target_layer(rois, gt_boxes, gt_ishard, dontcare_areas, self.n_classes)
            rois = roi_data[0]

        # endregion

        # project into feature maps
        face1, body1 = self.project_into_feature_maps(x1, rois, im_info)
        face2, body2 = self.project_into_feature_maps(x2, rois, im_info)
        face3, body3 = self.project_into_feature_maps(x3, rois, im_info)

        # TODO: add dropout layer?
        bodies = torch.cat((self.l2norm_bodies_1(self.roi_pool1(body1, rois)),
                            self.l2norm_bodies_2(self.roi_pool2(body2, rois)),
                            self.l2norm_bodies_3(self.roi_pool(body3, rois))), 0)
        # bodies = self.l2norm_fc_1(bodies)
        bodies = self.conv1x1_bodies(bodies)
        bodies = bodies.view(bodies.size()[0], -1)

        # TODO: add dropout layer?
        faces = torch.cat((self.l2norm_faces_1(self.roi_pool1(face1, rois)),
                           self.l2norm_faces_2(self.roi_pool2(face2, rois)),
                           self.l2norm_faces_3(face3)), 0)
        # faces = self.l2norm_fc_2(faces)
        faces = self.conv1x1_faces(faces)
        faces = faces.view(faces.size()[0], -1)

        # TODO: add dropout layer?
        temp = torch.cat((self.fc1(bodies), self.fc2(faces)), 0)
        scores = F.softmax(self.scores_fc(temp))
        bbox_pred = self.bbox_fc(temp)

        if self.training:
            self.cross_entropy, self.loss_box = self.build_loss(scores, bbox_pred, roi_data)

        return scores, bbox_pred, rois

    @property
    def loss(self):
        return self.cross_entropy + self.loss_box * 10

    def build_loss(self, cls_score, bbox_pred, roi_data):
        # classification loss
        label = roi_data[1].squeeze()
        fg_cnt = torch.sum(label.data.ne(0))
        bg_cnt = label.data.numel() - fg_cnt

        # for log
        if self.debug:
            maxv, predict = cls_score.data.max(1)
            self.tp = torch.sum(predict[:fg_cnt].eq(label.data[:fg_cnt])) if fg_cnt > 0 else 0
            self.tf = torch.sum(predict[fg_cnt:].eq(label.data[fg_cnt:]))
            self.fg_cnt = fg_cnt
            self.bg_cnt = bg_cnt

        ce_weights = torch.ones(cls_score.size()[1])
        ce_weights[0] = float(fg_cnt) / bg_cnt
        ce_weights = ce_weights.cuda()
        cross_entropy = F.cross_entropy(cls_score, label, weight=ce_weights)

        # bounding box regression L1 loss
        bbox_targets, bbox_inside_weights, bbox_outside_weights = roi_data[2:]
        bbox_targets = torch.mul(bbox_targets, bbox_inside_weights)
        bbox_pred = torch.mul(bbox_pred, bbox_inside_weights)

        loss_box = F.smooth_l1_loss(bbox_pred, bbox_targets, size_average=False) / (fg_cnt + 1e-4)

        return cross_entropy, loss_box

    @staticmethod
    def proposal_target_layer(rpn_rois, gt_boxes, gt_ishard, dontcare_areas, num_classes):
        """
        ----------
        rpn_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        gt_boxes: (G, 5) [x1 ,y1 ,x2, y2, class] int
        # gt_ishard: (G, 1) {0 | 1} 1 indicates hard
        dontcare_areas: (D, 4) [ x1, y1, x2, y2]
        num_classes
        ----------
        Returns
        ----------
        rois: (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        labels: (1 x H x W x A, 1) {0,1,...,_num_classes-1}
        bbox_targets: (1 x H x W x A, K x4) [dx1, dy1, dx2, dy2]
        bbox_inside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
        bbox_outside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
        """
        rpn_rois = rpn_rois.data.cpu().numpy()
        rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
            proposal_target_layer_py(rpn_rois, gt_boxes, gt_ishard, dontcare_areas, num_classes)
        # print labels.shape, bbox_targets.shape, bbox_inside_weights.shape
        rois = network.np_to_variable(rois, is_cuda=True)
        labels = network.np_to_variable(labels, is_cuda=True, dtype=torch.LongTensor)
        bbox_targets = network.np_to_variable(bbox_targets, is_cuda=True)
        bbox_inside_weights = network.np_to_variable(bbox_inside_weights, is_cuda=True)
        bbox_outside_weights = network.np_to_variable(bbox_outside_weights, is_cuda=True)

        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def interpret_faster_rcnn(self, cls_prob, bbox_pred, rois, im_info, im_shape, nms=True, clip=True, min_score=0.0):
        # find class
        scores, inds = cls_prob.data.max(1)
        scores, inds = scores.cpu().numpy(), inds.cpu().numpy()

        keep = np.where((inds > 0) & (scores >= min_score))
        scores, inds = scores[keep], inds[keep]

        # Apply bounding-box regression deltas
        keep = keep[0]
        box_deltas = bbox_pred.data.cpu().numpy()[keep]
        box_deltas = np.asarray([
            box_deltas[i, (inds[i] * 4): (inds[i] * 4 + 4)] for i in range(len(inds))
        ], dtype=np.float)
        boxes = rois.data.cpu().numpy()[keep, 1:5] / im_info[0][2]
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        if clip:
            pred_boxes = clip_boxes(pred_boxes, im_shape)

        # nms
        if nms and pred_boxes.shape[0] > 0:
            pred_boxes, scores, inds = nms_detections(pred_boxes, scores, 0.3, inds=inds)

        return pred_boxes, scores, self.classes[inds]

    def detect(self, image, thr=0.3):
        im_data, im_scales = self.get_image_blob(image)
        im_info = np.array(
            [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
            dtype=np.float32)

        cls_prob, bbox_pred, rois = self(im_data, im_info)
        pred_boxes, scores, classes = \
            self.interpret_faster_rcnn(cls_prob, bbox_pred, rois, im_info, image.shape, min_score=thr)
        return pred_boxes, scores, classes

    def get_image_blob_noscale(self, im):
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= self.PIXEL_MEANS

        processed_ims = [im]
        im_scale_factors = [1.0]

        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)

    def get_image_blob(self, im):
        """Converts an image into a network input.
        Arguments:
            im (ndarray): a color image in BGR order
        Returns:
            blob (ndarray): a data blob holding an image pyramid
            im_scale_factors (list): list of image scales (relative to im) used
                in the image pyramid
        """
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= self.PIXEL_MEANS

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in self.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > self.MAX_SIZE:
                im_scale = float(self.MAX_SIZE) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)

    def load_from_npz(self, params):
        self.rpn.load_from_npz(params)

        pairs = {'fc6.fc': 'fc6', 'fc7.fc': 'fc7', 'score_fc.fc': 'cls_score', 'bbox_fc.fc': 'bbox_pred'}
        own_dict = self.state_dict()
        for k, v in list(pairs.items()):
            key = '{}.weight'.format(k)
            param = torch.from_numpy(params['{}/weights:0'.format(v)]).permute(1, 0)
            own_dict[key].copy_(param)

            key = '{}.bias'.format(k)
            param = torch.from_numpy(params['{}/biases:0'.format(v)])
            own_dict[key].copy_(param)

    def load_from_rcnn(self, rcnn_model_path):
        detector = FasterRCNN()
        network.load_net(rcnn_model_path, detector)

        self.rpn = detector.rpn
        print(self.rpn.eval())

    @staticmethod
    def project_into_feature_maps(feature_maps, rois, im_info):
        """
        project RPN outputs on feature maps and return the head with the body
        tx = (xb-xf)/wf
        ty = (yb-yf)/hf
        tw = log(wb/wf)
        th = log(hb/hf)
        tx, ty, tw, and th are fixed
        x, y, h, w are center_x, center_y, height and width
        b, and f stand for body and face respectively
        :param feature_maps:
        :param rois: regions of interests
        :param im_info: image info [ width, height, scale ]
        :return: faces, bodies
        """

        tx, ty, tw, th = 1, 1, 1, 1

        _, _, w, h = feature_maps.size()
        _, bboxes = rois[0], rois[1:]

        # calculate the width and height of the face's bbox
        face_width = bboxes[:, 3] - bboxes[:, 1]
        face_height = bboxes[:, 4] - bboxes[:, 2]

        # center of the face
        face_center_x = bboxes[:, 1] + face_width
        face_center_y = bboxes[:, 2] + face_height

        im_info = im_info[0]
        # pooling size from the original image to current feature map
        r1 = im_info[1] / w
        r2 = im_info[0] / h

        # faces
        # bboxes[0::2] = max(min(bboxes[0::2] / r1, w), 0)
        # bboxes[1::2] = max(min(bboxes[1::2] / r2, h), 0)
        bboxes.data[:, 2::2] = bboxes.data[:, 2::2] / r1
        bboxes.data[:, 1::2] = bboxes.data[:, 1::2] / r2
        bboxes = bboxes.clamp(min=0)

        # bodies
        b_bboxes = torch.autograd.Variable(torch.zeros(bboxes.size()))

        # calculate body's center, width and height
        body_width = face_width * np.exp(tw)
        body_height = face_height * np.exp(th)
        body_center_x = face_width * tx + face_center_x
        body_center_y = face_height * ty + face_center_y

        # calculate bbox points
        b_bboxes[:, 1] = body_center_x - body_width / 2
        b_bboxes[:, 3] = body_center_x + body_width / 2

        b_bboxes[:, 2] = body_center_y - body_height / 2
        b_bboxes[:, 4] = body_center_y - body_height / 2
        b_bboxes = b_bboxes.clamp(min=0)

        # b_bboxes[0::2] = max(min(tx * face_width + face_center_x, w), 0)
        # b_bboxes[1::2] = max(min(ty * face_height + face_center_y, h), 0)

        # return feature_maps[:, bboxes[0]:bboxes[2], bboxes[1]: bboxes[3]], \
        #             feature_maps[:, b_bboxes[0]:b_bboxes[2],b_bboxes[1]: b_bboxes[3]]

        return bboxes, b_bboxes
