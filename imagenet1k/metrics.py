import torch

def correct_top_k(outputs, targets, top_k=(1,5)):
    """
    Find number of correct predictions for one batch.
    Args:
        outputs (torch.Tensor): Nx(class_number) Tensor containing logits.
        targets (torch.Tensor): N Tensor containing ground truths.
        top_k (Tuple): checking the ground truth is included in top-k prediction.
    Returns:
        List: List of number of top-1 and top-5 correct predictions.
    """
    # Our
    with torch.no_grad():
        prediction = torch.argsort(outputs, dim=-1, descending=True)
        result= []
        for k in top_k:
            correct_k = torch.sum((prediction[:, 0:k] == targets.unsqueeze(dim=-1)).any(dim=-1).float()).item() 
            result.append(correct_k)
        return result


# SimSiam
def accuracy(outputs, targets, top_k=(1,5)):
    with torch.no_grad():
        max_k = max(top_k)
        batch_size = targets.size(0)

        _, preds = outputs.topk(max_k, 1, True, True)
        preds = preds.t()
        correct = preds.eq(targets.view(1, -1).expand_as(preds))

        res = []
        for k in top_k:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



def grad_norm(model):
    with torch.no_grad():
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** (1. / 2)

