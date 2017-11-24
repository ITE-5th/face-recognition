import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class RoIPool(nn.Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(RoIPool, self).__init__()
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size()[0]
        outputs = Variable(torch.zeros(num_rois, num_channels, self.pooled_height, self.pooled_width)).cuda()

        for roi_ind, roi in enumerate(rois):
            batch_ind = int(roi[0].data[0])
            roi_start_w, roi_start_h, roi_end_w, roi_end_h = np.round(
                roi[1:].data.cpu().numpy() * self.spatial_scale).astype(int)
            roi_width = max(roi_end_w - roi_start_w + 1, 1)
            roi_height = max(roi_end_h - roi_start_h + 1, 1)
            bin_size_w = float(roi_width) / float(self.pooled_width)
            bin_size_h = float(roi_height) / float(self.pooled_height)

            for ph in range(self.pooled_height):
                h_start = int(np.floor(ph * bin_size_h))
                h_end = int(np.ceil((ph + 1) * bin_size_h))
                h_start = min(data_height, max(0, h_start + roi_start_h))
                h_end = min(data_height, max(0, h_end + roi_start_h))
                for pw in range(self.pooled_width):
                    w_start = int(np.floor(pw * bin_size_w))
                    w_end = int(np.ceil((pw + 1) * bin_size_w))
                    w_start = min(data_width, max(0, w_start + roi_start_w))
                    w_end = min(data_width, max(0, w_end + roi_start_w))

                    is_empty = (h_end <= h_start) or (w_end <= w_start)
                    if is_empty:
                        outputs[roi_ind, :, ph, pw] = 0
                    else:
                        data = features[batch_ind]
                        outputs[roi_ind, :, ph, pw] = torch.max(
                            torch.max(data[:, h_start:h_end, w_start:w_end], 1)[0], 2)[0].view(-1)

        return outputs
