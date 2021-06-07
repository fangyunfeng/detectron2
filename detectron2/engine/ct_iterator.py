from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F


def resize_ct(ct, out_size):
    n, h, w = ct.shape
    inv_trans = ct.new_tensor([[1, 0, 0], [0, 1, 0]])

    # out_size: h, w
    if torch.Size([h, w]) != torch.Size(out_size):
        batch_img = ct.unsqueeze_(dim=0)
        batch_img = F.interpolate(batch_img, size=out_size,
                                  mode='bilinear', align_corners=False)
        ct = batch_img.squeeze_(0)
        _T = ct.new_tensor([[w / out_size[1], 0., 0],
                            [0., h / out_size[0], 0],
                            [0., 0., 1.]])
        inv_trans = inv_trans.mm(_T)
    return ct, inv_trans


def resize_and_padding(batch, resize_size, target_dim):
    if isinstance(target_dim, (list, tuple)):
        if target_dim[0] != target_dim[1]:
            raise NotImplementedError("only support square")
        target_dim = target_dim[0]

    n = batch.size(0)
    new_h = resize_size
    new_w = resize_size
    resized, inv_trans = resize_ct(batch, (new_h, new_w))

    if new_h == new_w:
        return resized, inv_trans

    # compute padding
    start_x, start_y = int((target_dim - new_w) / 2), int((target_dim - new_h) / 2)
    out = batch.new_zeros((n, target_dim, target_dim))
    out[:, start_y:start_y + new_h, start_x:start_x + new_w] = resized

    # update inverse transform
    _T = batch.new_tensor([[1., 0., -start_x],
                           [0., 1., -start_y],
                           [0., 0., 1.]])
    inv_trans = inv_trans.mm(_T)
    return out, inv_trans


class CTIterator(object):
    def __init__(self, vol, out_size, device, indices=None, view="axial",
                 spacing=None, norm=False, in_channels=9, batch_size=8):
        view2axis = {"axial": 0, "coronal": 1, "sagittal": 2}
        if view not in view2axis:
            raise Exception("unknown view")

        self.idx = 0
        self.vol = vol
        self.norm = norm
        self.view = view
        self.device = device
        self.out_size = out_size
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.query_dim = view2axis[view]

        self.spacing = spacing
        self.original_size = tuple(vol.size(x) for x in set(range(3)) - {self.query_dim})
        self._compute_out_size()
        self.max_index = vol.size(self.query_dim)

        _k = in_channels // 2
        self._k = _k
        self.offset = torch.arange(-_k, _k + 1, dtype=torch.long, device=vol.device).unsqueeze(dim=0)
        # self.offset = torch.arange(-_k, _k + 1, dtype=torch.long, device=device).unsqueeze(dim=0)

        if indices is None:
            # indices = torch.arange(0, self.max_index, dtype=torch.long, device=self.device)
            indices = torch.arange(0, self.max_index, dtype=torch.long, device=vol.device)
        else:
            indices = indices.to(vol.device)

        self.indices = indices
        self.n_query_slices = self.indices.size(0)
        self.batch_shape = (-1, self.in_channels, out_size, out_size)

    def _compute_out_size(self):
        if self.spacing is None or self.view == "axial":
            self.resize_size = self.out_size
        else:
            sx, sy, sz = self.spacing
            sx = 0.5 * (sx + sy)

            h_hat, w_hat = self.original_size  # not really h and w, h always z
            if sx != sz:
                h_hat = int(sz / sx * h_hat)

            max_dim = max(h_hat, w_hat)
            scale = self.out_size[0] / max_dim
            self.resize_size = (int(round(scale * h_hat)), int(round(scale * w_hat)))

    def _preprocess_data(self, vol):
        vol = vol.to(self.device).type(torch.cuda.FloatTensor)
        vol, self.inv_trans = resize_and_padding(vol, self.resize_size, self.out_size)
        if self.view != "axial":
            vol = vol.clamp_(0., 255.).floor_()
        if self.norm:
            vol = vol.div_(255.)
        return vol

    def _index_vol(self, inds):
        vol = self.vol.index_select(self.query_dim, inds)  # not in place, safe
        if self.query_dim != 0:
            vol = vol.permute((self.query_dim, 0, 3 - self.query_dim))
            vol = vol.clamp_(0., 255.).floor_()
        return vol

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.idx >= self.n_query_slices:
            raise StopIteration
        else:
            start, stop = self.idx, min(self.idx + self.batch_size, self.n_query_slices)
            c_inds = self.indices[start: stop]
            inds = torch.flatten(c_inds.unsqueeze(dim=1) + self.offset).clamp_(min=0,
                                                                               max=self.max_index - 1)
            _vol = self._preprocess_data(self._index_vol(inds)).view(self.batch_shape)
            self.idx += self.batch_size
            return _vol
