import torch
import torch.nn.functional as F


def consistency_loss(logits, lbd, eta=0.5, loss='default'):
    """
    Consistency regularization for certified robustness.

    Parameters
    ----------
    logits : List[torch.Tensor]
        A list of logit batches of the same shape, where each
        is sampled from f(x + noise) with i.i.d. noises.
        len(logits) determines the number of noises, i.e., m > 1.
    lbd : float
        Hyperparameter that controls the strength of the regularization.
    eta : float (default: 0.5)
        Hyperparameter that controls the strength of the entropy term.
        Currently used only when loss='default'.
    loss : {'default', 'xent', 'kl', 'mse'} (optional)
        Which loss to minimize to obtain consistency.
        - 'default': The default form of loss.
            All the values in the paper are reproducible with this option.
            The form is equivalent to 'xent' when eta = lbd, but allows
            a larger lbd (e.g., lbd = 20) when eta is smaller (e.g., eta < 1).
        - 'xent': The cross-entropy loss.
            A special case of loss='default' when eta = lbd. One should use
            a lower lbd (e.g., lbd = 3) for better results.
        - 'kl': The KL-divergence between each predictions and their average.
        - 'mse': The mean-squared error between the first two predictions.

    """

    m = len(logits)
    softmax = [F.softmax(logit, dim=1) for logit in logits]
    avg_softmax = sum(softmax) / m

    loss_kl = [kl_div(logit, avg_softmax) for logit in logits]
    loss_kl = sum(loss_kl) / m

    if loss == 'default':
        loss_ent = entropy(avg_softmax)
        consistency = lbd * loss_kl + eta * loss_ent
    elif loss == 'xent':
        loss_ent = entropy(avg_softmax)
        consistency = lbd * (loss_kl + loss_ent)
    elif loss == 'kl':
        consistency = lbd * loss_kl
    elif loss == 'mse':
        sm1, sm2 = softmax[0], softmax[1]
        loss_mse = ((sm2 - sm1) ** 2).sum(1)
        consistency = lbd * loss_mse
    elif loss == 'jsd':
        log_avg_sm = torch.log(avg_softmax.clamp(min=1e-20))
        loss_ikl = [kl_div(log_avg_sm, F.softmax(logit, dim=1)) for logit in logits]
        loss_ikl = sum(loss_ikl) / m

        loss_jsd = 0.5 * (loss_kl + loss_ikl)
        consistency = lbd * loss_jsd
    elif loss == 'jsd2':
        def _jsd(p, q):
            r = (p + q) / 2.
            log_r = torch.log(r.clamp(min=1e-20))
            kl_p = F.kl_div(log_r, p, reduction='none').sum(1)
            kl_q = F.kl_div(log_r, q, reduction='none').sum(1)
            return (kl_p + kl_q) / 2.
        loss_jsd = [_jsd(avg_softmax, F.softmax(logit, dim=1)) for logit in logits]
        loss_jsd = sum(loss_jsd) / m
        consistency = lbd * loss_jsd
    else:
        raise NotImplementedError()

    return consistency.mean()


def kl_div(input, targets):
    return F.kl_div(F.log_softmax(input, dim=1), targets, reduction='none').sum(1)


def entropy(input):
    logsoftmax = torch.log(input.clamp(min=1e-20))
    xent = (-input * logsoftmax).sum(1)
    return xent
