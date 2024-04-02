import sys
sys.path.append('SegGPT/SegGPT_inference')

import torch as T
import torch.nn.functional as F
from SegGPT.SegGPT_inference.models_seggpt import SegGPT, seggpt_vit_large_patch16_input896x448
from typing import Tuple

class AdapterSegGPT(T.nn.Module):
    def __init__(self, seggpt_model: SegGPT, image_size: Tuple[int, int] = (448, 448)):
        super().__init__()

        self.seggpt = seggpt_model
        for param in self.seggpt.parameters():
            param.requires_grad = False

        self.image_size = image_size
        self.image_tensor = self.init_weight((1, 3, self.image_size[0], self.image_size[1]))
        self.prompt_tensor = self.init_weight((1, 3, self.image_size[0], self.image_size[1]))

    def init_weight(self, size: Tuple):
        weight = T.empty(size, requires_grad=True)
        weight = T.nn.init.kaiming_normal_(weight)
        return T.nn.Parameter(weight, requires_grad=True)

    def forward(self, imgs, tgts, bool_masked_pos=None, valid=None, seg_type=None, merge_between_batch=-1):
        # imgs, tgts N, 3, H, W
        B = imgs.shape[0]

        img_tensor = self.image_tensor.expand(B, -1, -1, -1)
        prompt_tensor = self.prompt_tensor.expand(B, -1, -1, -1)
        
        input_imgs = T.cat((img_tensor, imgs), dim=2)
        input_tgts = T.cat((prompt_tensor, tgts), dim=2)

        return self.seggpt.forward(input_imgs, input_tgts, bool_masked_pos, valid, seg_type, merge_between_batch)

if __name__ == '__main__':
    from data import BaseDataset

    dataset = BaseDataset('/home/steve/Datasets/OpenEarthMap-FSS/valset', is_train=True, class_idx=9)
    a, b, c, d, e,f,g = dataset[0]
    a=a
    b=b
    c=c
    d=d
    e=e

    model = seggpt_vit_large_patch16_input896x448()
    initial_ckpt = T.load('tuning2.pt', map_location='cpu')
    model.load_state_dict(initial_ckpt['model_state_dict'], strict=False)

    adapter = AdapterSegGPT(model)
    result = adapter(a.unsqueeze(0), b.unsqueeze(0), c.unsqueeze(0), d.unsqueeze(0), e.unsqueeze(0))