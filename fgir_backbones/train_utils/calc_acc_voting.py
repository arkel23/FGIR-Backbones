import torch
from einops import rearrange
from collections import Counter


def most_common_pred(preds_ind, voters=5, topk=5):
    # tensor shape: b, n, k where b is batch size, n is number of preds, k number of classes
    # voters is the number of top predictions to include in the majority vote
    b, n, k = preds_ind.shape
    preds_ind = rearrange(preds_ind, 'b n k -> (b n) k')

    assert k >= voters
    assert voters >= topk

    _, preds_ind = preds_ind.topk(voters, 1, True, True)

    preds_ind = rearrange(preds_ind, '(b n) voters -> b n voters', b=b, n=n)

    weighted_voters = int(voters * (voters + 1) / 2)
    ph = torch.empty(b, n, weighted_voters, device=preds_ind.device)

    for i in range(b):
        for j in range(n):
            repeats = torch.flip(torch.arange(1, voters+1, device=preds_ind.device), dims=[0])
            ph[i, j, :] = torch.repeat_interleave(preds_ind[i, j, :], repeats)

    preds_ind = rearrange(ph, 'b n weighted_voters -> b (n weighted_voters)')

    # preds_ind = rearrange(preds_ind, '(b n) voters -> b (n voters)', b=b, n=n)

    pred_list = preds_ind.tolist()
    # Use Counter to count occurrences of each integer
    counter_list = [Counter(preds) for preds in pred_list]

    # Find the most common integer for each
    most_common_list = [counter.most_common(topk) for counter in counter_list]
    if topk == 1:
        most_common = [common[0][0] for common in most_common_list]
        most_common = torch.tensor(most_common, device=preds_ind.device)
        most_common = rearrange(most_common, 'b -> 1 b')
    else:
        most_common_list_2 = []
        for k in range(topk):
            most_common_list_2.append([common[k][0] for common in most_common_list])
        most_common = torch.tensor(most_common_list_2, device=preds_ind.device)

    return most_common


def accuracy_vote(preds_ind, target, voters=5, topk=(1,)):
    with torch.no_grad():
        batch_size = target.size(0)
        maxk = max(topk)
        preds = most_common_pred(preds_ind, voters, maxk)

        correct = preds.eq(target.reshape(1, -1).expand_as(preds))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
