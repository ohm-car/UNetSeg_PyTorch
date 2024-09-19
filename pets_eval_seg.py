import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils.percLoss import percLoss
from dice_loss import dice_coeff
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.functional.classification import binary_jaccard_index


def eval_net(net, loader, device, regularizer):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    # recon_loss = 0
    mask_loss = 0
    iou = 0
    tot = 0
    iou_metric = BinaryJaccardIndex()

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, recon_img, true_masks, true_perc = batch['image'], batch['reconstructed_image'], batch['mask'], batch['mask_perc']
            imgs = imgs.to(device=device, dtype=torch.float32)
            print(true_masks.shape)
            true_masks = true_masks.to(device=device, dtype=torch.long)
            # true_masks = true_masks.to(device=device, dtype=torch.float32)
            recon_img = recon_img.to(device=device, dtype=torch.float32)
            true_perc = true_perc.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                pred_mask = net(imgs)

            if net.n_classes > 1:
                # recon_loss_batch = F.l1_loss(pred_recon_img, recon_img).item()
                criterion = nn.CrossEntropyLoss(reduction='mean')
                mask_loss_batch = criterion(pred_mask, true_masks).item()

                mean_batch_iou = 0
                for i in range(len(pred_mask)):
                    single_iou = binary_jaccard_index(pred_mask[i], true_masks[i])
                    # print(single_iou)
                    mean_batch_iou += single_iou
                mean_batch_iou = mean_batch_iou / len(pred_mask)

                batch_iou = binary_jaccard_index(pred_mask, true_masks)
                # print(batch_iou, mean_batch_iou / len(pred_mask), mean_batch_iou)

                print(mean_batch_iou, batch_iou)
                iou += mean_batch_iou
                # recon_loss += recon_loss_batch
                mask_loss += mask_loss_batch
                # tot += mask_loss_batch
            else:
                # recon_loss_batch = F.l1_loss(pred_recon_img, recon_img).item()
                criterion = nn.CrossEntropyLoss(reduction='mean')
                mask_loss_batch = criterion(pred_mask, true_masks).item()

                mean_batch_iou = 0
                for i in range(len(pred_mask)):
                    single_iou = binary_jaccard_index(pred_mask[i], true_masks[i])
                    # print(single_iou)
                    mean_batch_iou += single_iou
                mean_batch_iou = mean_batch_iou / len(pred_mask)

                batch_iou = binary_jaccard_index(pred_mask, true_masks)
                # print(batch_iou, mean_batch_iou / len(pred_mask), mean_batch_iou)

                print(mean_batch_iou, batch_iou)
                iou += mean_batch_iou
                # recon_loss += recon_loss_batch
                mask_loss += mask_loss_batch
                # tot += mask_loss_batch
            pbar.update()

    net.train()
    print("Val IOU: ", iou / n_val)
    return mask_loss / n_val, iou / n_val
