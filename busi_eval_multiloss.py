import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.percLoss import percLoss
from dice_loss import dice_coeff
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.functional.classification import binary_jaccard_index, multiclass_jaccard_index
from torchmetrics.functional.segmentation import mean_iou


def eval_net(net, loader, device, regularizer, epoch):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    # mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    # print("n_val: ", n_val)
    seg_loss = 0
    mask_loss = 0
    iou = 0
    tot = 0
    iou_metric = BinaryJaccardIndex()

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, recon_imgs, true_masks, true_perc = batch['image'], batch['reconstructed_image'], batch['mask'], batch['mask_perc']
            imgs = imgs.to(device=device, dtype=torch.float32)
            # true_masks = true_masks.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)
            # true_masks = torch.unsqueeze(true_masks, 1)
            # print(true_masks.shape)
            # amax_true_masks = torch.argmax(true_masks, dim=1)
            # print(amax_true_masks.shape)
            recon_imgs = recon_imgs.to(device=device, dtype=torch.float32)
            true_perc = true_perc.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                outs = net(imgs)
                pred_masks, pred_imgs = outs['out'], outs['aux']
                # pred_masks, pred_imgs = outs[1], outs[0]
                # pred_masks = torch.squeeze(pred_masks)
                # pred_masks = F.softmax(pred_masks, dim=1)
                # print("Predictions Shape: ", pred_masks.shape)
                # print("Targets Shape: ", true_masks.shape)

            # if net.n_classes > 1:
            if True:
                seg_loss_batch = F.l1_loss(pred_imgs, recon_imgs).item()
                # seg_loss_batch = F.cross_entropy(pred_masks, true_masks).item()
                pcLossCriterion = percLoss(threshold_prob = 0.9, regularizer = regularizer)
                mask_loss_batch = pcLossCriterion(pred_masks, true_perc).item()

                # mean_batch_iou = 0
                # for i in range(len(pred_masks)):
                #     # print("pred_masks[i] shape: ", pred_masks[i].shape)
                #     # print("targets[i] shape: ", true_masks[i].shape)
                #     single_iou = multiclass_jaccard_index(pred_masks[i], true_masks[i], num_classes=20)
                #     # print(single_iou)
                #     mean_batch_iou += single_iou

                # batch_iou = multiclass_jaccard_index(pred_masks, amax_true_masks, num_classes=2)
                # print("Sanity Check: ", torch.sum(pred_masks, dim=1) == 1)
                # class_mIU = mean_iou(torch.argmax(pred_masks, dim=1), true_masks, num_classes=2, per_class=True)
                # print("Class_mIU shape: ", class_mIU.shape)
                # class_bmIU = torch.mean(class_mIU, 0)
                # print("Mean IoU per class: ", class_bmIU, class_bmIU.shape)
                # mIU = mean_iou(torch.argmax(pred_masks, dim=1), true_masks, num_classes=2, per_class=False)
                # bmIU = torch.mean(mIU, 0)
                # print("Mean IoU overall: ", bmIU, bmIU.shape)
                # test_iou = multiclass_jaccard_index(pred_masks, true_masks, num_classes=2, average=None)
                # print("Test IoU: ", test_iou)
                # batch_iou = multiclass_jaccard_index(pred_masks, true_masks, num_classes=2)

                mean_batch_iou = 0
                for i in range(len(pred_masks)):
                    single_iou = binary_jaccard_index(pred_masks[i], true_masks[i])
                    # print(single_iou)
                    mean_batch_iou += single_iou

                batch_iou = binary_jaccard_index(pred_masks, true_masks)
                # print("Overall batch IoU: ", batch_iou)
                # print("Mean of Test IoU: ", torch.mean(test_iou))
                # print(batch_iou, mean_batch_iou / len(pred_masks), mean_batch_iou)

                iou += (mean_batch_iou / len(pred_masks))
                # iou += mean_batch_iou
                seg_loss += seg_loss_batch
                mask_loss += mask_loss_batch
                tot += seg_loss_batch + mask_loss_batch
                # tot += seg_loss_batch
            else:
                seg_loss_batch = F.l1_loss(pred_imgs, recon_imgs).item()
                # seg_loss_batch = F.cross_entropy(pred_masks, true_masks).item()
                # pcLossCriterion = percLoss(threshold_prob = 0.9, regularizer = regularizer)
                # mask_loss_batch = pcLossCriterion(pred_mask, true_perc).item()

                # mean_batch_iou = 0
                # for i in range(len(pred_masks)):
                #     single_iou = multiclass_jaccard_index(pred_masks[i], true_masks[i], num_classes=21)
                #     # print(single_iou)
                #     mean_batch_iou += single_iou

                batch_iou = multiclass_jaccard_index(pred_masks, true_masks, num_classes=21)
                # print(batch_iou, mean_batch_iou / len(pred_mask), mean_batch_iou)

                # iou += (mean_batch_iou / len(pred_masks))
                iou += batch_iou
                seg_loss += seg_loss_batch
                # mask_loss += mask_loss_batch
                # tot += seg_loss_batch + mask_loss_batch
                tot += seg_loss_batch
            pbar.update()

    net.train()
    n_val = max(n_val, 1)
    print("Epoch: ", epoch, "Val IOU: ", iou / n_val)
    return tot / n_val, mask_loss / n_val, seg_loss / n_val, iou / n_val
