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
    with torch.no_grad():
        prediction = torch.argsort(outputs, dim=-1, descending=True)
        result= []
        for k in top_k:
            correct_k = torch.sum((prediction[:, 0:k] == targets.unsqueeze(dim=-1)).any(dim=-1).float()).item() 
            result.append(correct_k)
        return result


def grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** (1. / 2)

