import torch
import torch.nn.functional as F

def dice_loss(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.long()]               
        true_1_hot = true_1_hot.permute(0, 3, 1, 2)                # (B, H, W, C) to (B, C, H, W)
        probas = F.softmax(logits, dim=1)
        
    true_1_hot = true_1_hot.type(logits.type()).contiguous()
    dims = (0,) + tuple(range(2, true.ndimension()))        # dims = (0, 2, 3)
    intersection = torch.sum(probas * true_1_hot, dims)     # intersection w.r.t. the class
    cardinality = torch.sum(probas + true_1_hot, dims)      # cardinality w.r.t. the class
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


def fitness_test(preds,targets):
    #IoU
    intersection=(preds & targets).float().sum((1,2))
    union=(preds | targets).float().sum((1,2))
    iou=(intersection+1e-6)/(union+1e-6)
    pix_accuracy=(preds==targets).float().sum((1,2))/preds[0].numel()
    return iou, pix_accuracy