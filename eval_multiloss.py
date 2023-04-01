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
            imgs, true_imrecon, true_perc = batch['image'], batch['reconstructed_image'], batch['mask_perc']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_imrecon = true_imrecon.to(device=device, dtype=mask_type)
            true_perc = true_perc.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                imrecon_pred, mask_pred = net(imgs)

            if net.n_classes > 1:
                pred_perc = torch.mean(torch.squeeze(mask_pred), (1,2))
                pred_perc = torch.unsqueeze(pred_perc, 1)
                tot += F.l1_loss(imrecon_pred, true_imrecon).item() + F.l1_loss(pred_perc, true_perc).item()
            else:
                pred = torch.sigmoid(imrecon_pred)
                pred = (pred > 0.5).float()
                # pred_perc = torch.mean(torch.squeeze(mask_pred), (1,2))
                # pred_perc = torch.unsqueeze(pred_perc, 1)
                # tot += 0.5 * (dice_coeff(pred, true_imrecon).item() + 100 * (1 - (torch.abs(pred_perc - true_perc) / true_perc)))
                tot += 0.5 * (dice_coeff(pred, true_imrecon).item())
            pbar.update()

    net.train()
    return tot / n_val
