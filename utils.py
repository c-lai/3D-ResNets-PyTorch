import csv
import random
from functools import partialmethod

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, \
    precision_recall_curve, roc_auc_score, \
        balanced_accuracy_score, accuracy_score, roc_curve, auc


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


class Logger(object):

    def __init__(self, path, header):
        self.log_file = path.open('w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def calculate_accuracy(outputs, targets):
    with torch.no_grad():
        batch_size = targets.size(0)

        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()

        return n_correct_elems / batch_size


def calculate_accuracy_binary(outputs, targets, threshold=0.5, balanced=False):
    with torch.no_grad():
        pred = torch.where(outputs>threshold, 1, 0)
        # pred = pred.t()
        if balanced:
            acc = balanced_accuracy_score(targets.view(-1, 1).cpu().numpy(),
                pred.cpu().numpy())
        else:
            acc = accuracy_score(targets.view(-1, 1).cpu().numpy(),
                pred.cpu().numpy())

        return acc


def calculate_precision_and_recall(outputs, targets, pos_label=1):
    with torch.no_grad():
        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets.view(-1, 1).cpu().numpy(),
            pred.cpu().numpy())

        return precision[pos_label], recall[pos_label], f1[pos_label]


def calculate_precision_and_recall_binary(outputs, targets, pos_label=1):
    with torch.no_grad():
        precisions, recalls, thresholds = precision_recall_curve(
            targets.view(-1, 1).cpu().numpy(), 
            outputs.cpu().numpy(),
            pos_label=pos_label)
        f1s = 2*precisions*recalls/(precisions+recalls+0.001)
        optimal_index = np.argmax(f1s)
        precision = precisions[optimal_index]
        recall = recalls[optimal_index]
        f1 = f1s[optimal_index]
        threshold = thresholds[optimal_index]
        # pred = torch.where(outputs>0.5, 1, 0)
        # precision, recall, f1, _ = precision_recall_fscore_support(
        #     targets.view(-1, 1).cpu().numpy(),
        #     pred.cpu().numpy(),
        #     pos_label=pos_label,
        #     average='binary',
        #     zero_division=0)

        return precision, recall, f1, threshold


def calculate_auc(outputs, targets, pos_label=1):
    with torch.no_grad():
        auc = roc_auc_score(
            targets.view(-1, 1).cpu().numpy(),
            outputs.cpu().numpy())

        return auc


def plot_roc(outputs, targets, filename='roc.png'):
    with torch.no_grad():
        fpr, tpr, _ = roc_curve(targets.view(-1, 1).cpu().numpy(),
            outputs.cpu().numpy())
        roc_auc = auc(fpr, tpr)
        
        import matplotlib.pyplot as plt
        plt.figure()
        lw = 2
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=lw,
            label="ROC curve (area = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="black", lw=lw, linestyle="--")
        plt.axis('square')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.0])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic curve")
        plt.legend(loc="lower right")
        plt.savefig(filename)


def save_roc_data(outputs, targets, filename):
    with torch.no_grad():
        fpr, tpr, _ = roc_curve(targets.view(-1, 1).cpu().numpy(),
            outputs.cpu().numpy())
        fprtpr = np.concatenate([fpr[None,:],tpr[None,:]])
        np.save(filename, fprtpr)


def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()

    random.seed(torch_seed + worker_id)

    if torch_seed >= 2**32:
        torch_seed = torch_seed % 2**32
    np.random.seed(torch_seed + worker_id)


def get_lr(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lr = float(param_group['lr'])
        lrs.append(lr)

    return max(lrs)


def partialclass(cls, *args, **kwargs):

    class PartialClass(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    return PartialClass


def select_n_random(dataset, n=50):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    perm = np.random.RandomState(seed=42).permutation(len(dataset))
    return torch.utils.data.Subset(dataset, perm[:n])


def get_activation(activation, name):
    def hook(module, input, output):
        activation[name] = output.detach()
    return hook
