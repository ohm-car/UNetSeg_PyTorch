import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.percLoss import percLoss
from dice_loss import dice_coeff


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, recon_img, true_mask = batch['image'], batch['reconstructed_image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            recon_img = recon_img.to(device=device, dtype=mask_type)
            true_mask = true_mask.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                pred_recon_img, pred_mask = net(imgs)

            if net.n_classes > 1:
                # pred_perc = torch.mean(torch.squeeze(pred_mask), (1,2))
                # pred_perc = torch.unsqueeze(pred_perc, 1)
                tot += F.l1_loss(pred_recon_img, recon_img).item() + F.l1_loss(pred_mask, true_mask).item()
            else:
                pred = torch.sigmoid(pred_recon_img)
                pred = (pred > 0.5).float()
                # pred_perc = torch.mean(torch.squeeze(pred_mask), (1,2))
                # pred_perc = torch.unsqueeze(pred_perc, 1)
                # tot += 0.5 * (dice_coeff(pred, recon_img).item() + 100 * (1 - (torch.abs(pred_perc - true_mask) / true_mask)))
                tot += 0.5 * (dice_coeff(pred, recon_img).item())
            pbar.update()

    net.train()
    return tot / n_val
