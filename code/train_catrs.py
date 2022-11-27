# this file is based on code publicly available at
#   https://github.com/locuslab/smoothing
# written by Jeremy Cohen.

import argparse
import time
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.distributions.binomial import Binomial

from architectures import ARCHITECTURES
from datasets import get_dataset, DATASETS
from train_utils import AverageMeter, accuracy, log, test, requires_grad_
from train_utils import prologue

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--noise_sd', default=0.0, type=float,
                    help="standard deviation of Gaussian noise for data augmentation")
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--id', default=None, type=int,
                    help='experiment id, `randint(10000)` if None')

#####################
# Options added by Salman et al. (2019)
parser.add_argument('--resume', action='store_true',
                    help='if true, tries to resume training from existing checkpoint')
parser.add_argument('--pretrained-model', type=str, default='',
                    help='Path to a pretrained model')

parser.add_argument('--num-noise-vec', default=4, type=int,
                    help="number of noise vectors. `m` in the paper.")

parser.add_argument('--epsilon', default=256, type=float,
                    help="radius of PGD (Projected Gradient Descent) attack")
parser.add_argument('--num-steps', default=4, type=int,
                    help="rumber of steps of PGD (Projected Gradient Descent) attack")
parser.add_argument('--lbd', default=1.0, type=float,
                    help="strength of the contribution of L^high")
parser.add_argument('--confidence_mask', action='store_true',
                    help='if true, choose K based on confidence of Gaussian (Cohen et al., 2019) baseline (mainly to bypass cold-start problem)')
parser.add_argument('--lr_drop', default=10000, type=int,
                    help='drops learning rate for given epoch')
parser.add_argument('--warmup', default=10000, type=int,
                    help='after given epoch, raise the attack radius by 2')

args = parser.parse_args()
args.outdir = f"logs/{args.dataset}/catrs/adv_{args.epsilon}_{args.num_steps}/lbd_{args.lbd}/num_{args.num_noise_vec}/noise_{args.noise_sd}"

args.epsilon /= 256.0


def main():
    train_loader, test_loader, criterion, model, optimizer, scheduler, \
    starting_epoch, logfilename, model_path, device, writer = prologue(args)

    pin_memory = (args.dataset == "imagenet")
    train_dataset = get_dataset(args.dataset, f'train_t{args.noise_sd}')
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                              num_workers=args.workers, pin_memory=pin_memory)

    attacker = KL_PGD(steps=args.num_steps, device=device, max_norm=args.epsilon)

    for epoch in range(starting_epoch, args.epochs):
        before = time.time()
        if epoch >= args.warmup:
            attacker = KL_PGD(steps=args.num_steps, device=device, max_norm=args.epsilon*2.0)
        
        if epoch == args.lr_drop:
            for g in optim.param_groups:
                g['lr'] = g['lr'] * args.gamma

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch,
                                      args.noise_sd, attacker, device, writer)
        test_loss, test_acc = test(test_loader, model, criterion, epoch, args.noise_sd, device, writer, args.print_freq)
        after = time.time()

        log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
            epoch, after - before,
            scheduler.get_lr()[0], train_loss, train_acc, test_loss, test_acc))

        # In PyTorch 1.1.0 and later, you should call `optimizer.step()` before `lr_scheduler.step()`.
        # See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        scheduler.step(epoch)

        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, model_path)


def _chunk_minibatch(batch, num_batches):
    X, y, pb = batch
    batch_size = len(X) // num_batches
    for i in range(num_batches):
        yield X[i*batch_size : (i+1)*batch_size], y[i*batch_size : (i+1)*batch_size], \
              pb[i*batch_size : (i+1)*batch_size]


def train(loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer,
          epoch: int, noise_sd: float, attacker, device: torch.device, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_reg = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to train mode
    model.train()
    _tril = torch.ones(args.num_noise_vec + 1, args.num_noise_vec).tril(-1).to(device)

    for i, batch in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        mini_batches = _chunk_minibatch(batch, args.num_noise_vec)
        for inputs, targets, pb in mini_batches:
            inputs, targets, pb = inputs.to(device), targets.to(device), pb.to(device)
            batch_size = inputs.size(0)

            noises = [torch.randn_like(inputs, device=device) * noise_sd
                      for _ in range(args.num_noise_vec)]
            noises0 = noises

            # augment inputs with noise
            inputs0_c = torch.cat([inputs + noise0 for noise0 in noises0], dim=0)
            targets_c = targets.repeat(args.num_noise_vec)

            # compute output
            logits0_c = model(inputs0_c)
            logits0_chunk = torch.chunk(logits0_c, args.num_noise_vec, dim=0)
            
            n_classes = logits0_c.size(1)

            t_sm0 = [F.one_hot(torch.argmax(logit, dim=1), n_classes) for logit in logits0_chunk]

            t_sm = sum(t_sm0) / args.num_noise_vec


            requires_grad_(model, False)
            model.eval()
            noises = attacker.attack(model, inputs, pb, noises=noises)
            model.train()
            requires_grad_(model, True)

            inputs_c = torch.cat([inputs + noise for noise in noises], dim=0)
            logits_c = model(inputs_c)

            loss_xent = [F.cross_entropy(logit, targets, reduction='none').view(-1, 1) for logit in logits0_chunk]
            loss_xent = torch.cat(loss_xent, dim=1)
            loss_xent, _ = loss_xent.sort()
            if args.confidence_mask:
                w = -F.nll_loss(pb, targets, reduction='none')
            else:
                w = -F.nll_loss(t_sm, targets, reduction='none')
            accept = Binomial(args.num_noise_vec, w).sample().long()
            accept = accept.clamp(min=1)
            mask_fa = (accept == args.num_noise_vec)

            loss_xent = (loss_xent * _tril[accept]).mean(1)

            logits_chunk = torch.chunk(logits_c, args.num_noise_vec, dim=0)
            loss_kl = [F.kl_div(F.log_softmax(logit, dim=1), pb, reduction='none').sum(1, keepdim=True)
                       for logit in logits_chunk]
            loss_klw, _ = torch.cat(loss_kl, dim=1).max(1)

            loss = (loss_xent + args.lbd * loss_klw * mask_fa).mean()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(logits_c, targets_c, topk=(1, 5))
            losses.update(loss_xent.mean().item(), batch_size)
            losses_reg.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    if writer:
        writer.add_scalar('loss/train', losses.avg, epoch)
        writer.add_scalar('loss/consistency', losses_reg.avg, epoch)
        writer.add_scalar('batch_time', batch_time.avg, epoch)
        writer.add_scalar('accuracy/train@1', top1.avg, epoch)
        writer.add_scalar('accuracy/train@5', top5.avg, epoch)

    return (losses.avg, top1.avg)


class KL_PGD(object):
    """
    SmoothAdv PGD L2 attack

    Parameters
    ----------
    steps : int
        Number of steps for the optimization.
    max_norm : float or None, optional
        If specified, the norms of the perturbations will not be greater than this value which might lower success rate.
    device : torch.device, optional
        Device on which to perform the attack.

    """

    def __init__(self,
                 steps: int,
                 random_start: bool = True,
                 max_norm: Optional[float] = None,
                 device: torch.device = torch.device('cpu')) -> None:
        super(KL_PGD, self).__init__()
        self.steps = steps
        self.random_start = random_start
        self.max_norm = max_norm
        self.device = device

    def attack(self, model, inputs, labels, noises=None):
        """
        Performs SmoothAdv PGD L2 attack of the model for the inputs and labels.

        Parameters
        ----------
        model : nn.Module
            Model to attack.
        inputs : torch.Tensor
            Batch of samples to attack. Values should be in the [0, 1] range.
        labels : torch.Tensor
            Labels of the samples to attack.
        noises : List[torch.Tensor]
            Lists of noise samples to attack.

        Returns
        -------
        torch.Tensor
            Batch of samples modified to be adversarial to the model.

        """
        if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')

        def _batch_l2norm(x):
            x_flat = x.reshape(x.size(0), -1)
            return torch.norm(x_flat, dim=1)

        m = len(noises)
        inputs_r = inputs.repeat(m, 1, 1, 1)
        noise0 = torch.cat(noises, dim=0)
        noise = noise0.detach()
        batch_size = inputs_r.size(0)

        mu0 = noise0.view(batch_size, -1).mean(1).view(-1, 1, 1, 1)
        sigma0 = (noise0 ** 2).view(batch_size, -1).mean(1).sqrt().view(-1, 1, 1, 1)

        alpha = self.max_norm / self.steps * 2
        for i in range(self.steps):
            noise.requires_grad_()
            logits_r = model(inputs_r + noise)

            logits_chunk = torch.chunk(logits_r, m, dim=0)
            loss_kls = [F.kl_div(F.log_softmax(logit, dim=1), labels, reduction='none').sum(1)
                       for logit in logits_chunk]
            loss = (sum(loss_kls) / m).sum()

            grad = torch.autograd.grad(loss, [noise])[0]
            grad_norm = _batch_l2norm(grad).view(-1, 1, 1, 1)
            grad = grad / (grad_norm + 1e-8)

            noise = noise + alpha * grad
            eta = noise - noise0
            eta = eta.renorm(p=2, dim=0, maxnorm=self.max_norm)

            noise = noise0 + eta

            mu = noise.view(batch_size, -1).mean(1).view(-1, 1, 1, 1)
            sigma = (noise ** 2).view(inputs_r.size(0), -1).mean(1).sqrt().view(-1, 1, 1, 1)

            noise = (noise - mu) / sigma
            noise = mu0 + noise * sigma0
            noise = noise.detach()

        return torch.chunk(noise, m, dim=0)


if __name__ == "__main__":
    main()
