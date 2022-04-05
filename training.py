import torch
import time
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

from utils import AverageMeter, calculate_accuracy, calculate_accuracy_binary, \
    calculate_precision_and_recall_binary, calculate_auc, \
        get_activation


def train_epoch(epoch,
                data_loader,
                subset_loader, 
                model,
                criterion,
                optimizer,
                device,
                current_lr,
                epoch_logger,
                batch_logger,
                tb_writer=None,
                distributed=False):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    output_list = []
    targets_list = []
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        targets = targets.to(device, non_blocking=True).view(-1, 1).float()
        targets_list.append(targets)
        NN_output = model(inputs)
        output = torch.sigmoid(NN_output)
        output_list.append(output)
        loss = criterion(NN_output, targets)
        acc = calculate_accuracy_binary(output, targets, balanced=True)

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        # precisions.update(precision, inputs.size(0))
        # recalls.update(recall, inputs.size(0))
        # f1s.update(f1, inputs.size(0))
        # aucs.update(auc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if batch_logger is not None:
            batch_logger.log({
                'epoch': epoch,
                'batch': i + 1,
                'iter': (epoch - 1) * len(data_loader) + (i + 1),
                'loss': losses.val,
                'acc': accuracies.val,
                # 'precision': precisions.val,
                # 'recall': recalls.val,
                # 'f1': f1s.val,
                # 'auc': aucs.val,
                'lr': current_lr
            })

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})\t'.format(epoch,
            #   'Precision {pre.val:.3f} ({pre.avg:.3f})\t'
            #   'Recall {rec.val:.3f} ({rec.avg:.3f})\t'
            #   'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
            #   'AUC {auc.val:.3f} ({auc.avg:.3f})\t'.format(epoch,
                                                         i + 1,
                                                         len(data_loader),
                                                         batch_time=batch_time,
                                                         data_time=data_time,
                                                         loss=losses,
                                                         acc=accuracies))
                                                        #  pre=precisions,
                                                        #  rec=recalls,
                                                        #  f1=f1s,
                                                        #  auc=aucs))
    outputs = torch.cat(output_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    precision, recall, f1, threshold= calculate_precision_and_recall_binary(outputs, targets)
    auc = calculate_auc(outputs, targets)
    acc = calculate_accuracy_binary(outputs, targets, threshold, balanced=True)
    loss = criterion(outputs, targets)

    if distributed:
        loss_sum = torch.tensor([losses.sum],
                                dtype=torch.float32,
                                device=device)
        loss_count = torch.tensor([losses.count],
                                  dtype=torch.float32,
                                  device=device)
        acc_sum = torch.tensor([accuracies.sum],
                               dtype=torch.float32,
                               device=device)
        acc_count = torch.tensor([accuracies.count],
                                 dtype=torch.float32,
                                 device=device)

        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_count, op=dist.ReduceOp.SUM)

        losses.avg = loss_sum.item() / loss_count.item()
        accuracies.avg = acc_sum.item() / acc_count.item()

    if epoch_logger is not None:
        epoch_logger.log({
            'epoch': epoch,
            'loss': loss.item(),
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'threshold': threshold,
            'lr': current_lr
        })

    if tb_writer is not None:
        tb_writer.add_scalar('train/loss', loss.item(), epoch)
        tb_writer.add_scalar('train/acc', acc, epoch)
        tb_writer.add_scalar('train/precision', precision, epoch)
        tb_writer.add_scalar('train/recall', recall, epoch)
        tb_writer.add_scalar('train/f1', f1, epoch)
        tb_writer.add_scalar('train/auc', auc, epoch)
        tb_writer.add_scalar('train/threshold', threshold, epoch)
        tb_writer.add_scalar('train/lr', current_lr, epoch)

        if not epoch%10:
            latent_vectors_list = []
            targets_list = []
            for i, (inputs, targets) in enumerate(subset_loader):
                activations = {}
                model.module.fc.register_forward_hook(get_activation(activations, 'fc'))
                outputs = model(inputs)

                latent_vectors_list.append(activations['fc'])
                targets_list.append(targets)
            latent_vectors = torch.cat(latent_vectors_list, dim=0)
            targets_subset = torch.cat(targets_list, dim=0)
            # features = latent_vectors.view(-1, 16)
            tb_writer.add_embedding(latent_vectors,
                                    metadata=targets_subset,
                                    global_step=epoch,
                                    tag='train/latent space')
