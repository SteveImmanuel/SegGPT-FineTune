import logging
import torch
import numpy as np

def setup_logging():
    logging.basicConfig(level=logging.INFO)

def get_logger(name: str, rank: int):
    # adapted from https://discuss.pytorch.org/t/ddp-training-log-issue/125808
    class NoOp:
        def __getattr__(self, *args):
            def no_op(*args, **kwargs):
                """Accept every signature by doing non-operation."""
                pass
            return no_op

    if rank == 0:
        return logging.getLogger(name)
    return NoOp()

def cmap_to_lbl(cmap: torch.Tensor, color_palette: torch.Tensor):
    B, H, W, C = cmap.shape
    _, N, _ = color_palette.shape
    
    dist_mat = torch.cdist(cmap.reshape(B, H * W, C), color_palette, p=2)
    dist_mat = dist_mat.reshape(B, H, W, N)
    label = torch.argmin(dist_mat, axis=3)

    result = torch.zeros_like(cmap)
    for i in range(B):
        for j in range(N):
            result[i][label[i] == j] = color_palette[i][j]

    return result, label

def calculate_iou(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, total_classes: int):
    #total class includes background
    result = torch.zeros((total_classes, 2), dtype=pred.dtype, device=pred.device)
    masked_gt = mask * gt
    masked_pred = mask * pred
    for i in range(total_classes):
        pred_total = (masked_pred == i)
        gt_total = (masked_gt == i)
        intersection = (pred_total & gt_total).sum()
        union = pred_total.sum() + gt_total.sum() - intersection
        result[i][0] += intersection
        result[i][1] += union
    return result

def create_stitch_mask(h, w, type, width):
    prompt_mask = np.zeros(h * w)
    image_mask = np.zeros((h, w))
    if type == 0:
        image_mask[:, image_mask.shape[1] // 2 - width: image_mask.shape[1] // 2 + width] = 1
    elif type == 1:
        image_mask[image_mask.shape[0] // 2 - width: image_mask.shape[0] // 2 + width, :] = 1
    else:
        image_mask[image_mask.shape[0] // 2 - width: image_mask.shape[0] // 2 + width, image_mask.shape[1] // 2 - width: image_mask.shape[1] // 2 + width] = 1
    image_mask = image_mask.flatten()
    result = np.concatenate((prompt_mask, image_mask))
    return result

if __name__ == '__main__':
    pred = torch.ones((5, 10, 10))
    gt = torch.ones((5, 10, 10))
    mask = torch.ones((5, 10, 10))
    res = calculate_iou(pred, gt, mask, 4)
    # cmap = torch.randn(10, 500, 400, 3)
    # cp = torch.randn(10, 20, 3)
    # cmap_to_lbl(cmap, cp)
    print(res)