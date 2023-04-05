import math
import time
from collections import OrderedDict
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import *
import utils

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, args, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, saver=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False):

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train(True)
    optimizer.zero_grad()

    end = time.time()
    last_idx = len(data_loader) - 1

    for data_iter_step, (samples, targets) in enumerate(data_loader):
        last_batch = data_iter_step == last_idx
        data_time_m.update(time.time() - end)
        step = data_iter_step // update_freq

        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    if "flag" in param_group.keys():  # radius specific learning rate!
                        param_group["lr"] = 0.1 * lr_schedule_values[it] * param_group["lr_scale"]
                    else:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets)
        else: # full precision
            output = model(samples)
            loss = criterion(output, targets)

        loss_value = loss.item()
        if not args.distributed:
            losses_m.update(loss.item(), samples.size(0))

        if not math.isfinite(loss_value): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        if args.distributed:
            if hasattr(model.module, 'clipping'):
                model.module.clipping()
        else:
            if hasattr(model, 'clipping'):
                model.clipping()
        torch.cuda.synchronize()

        batch_time_m.update(time.time() - end)
        if last_batch or data_iter_step % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item() * update_freq, samples.size(0))  
            
            print('Train: {} [{:>4d}/{} ({:>3.0f}%)] ' \
                  'Loss: {loss.val:#.4g} ({loss.avg:#.3g}) ' \
                  'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ' \
                  '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) ' \
                  'LR: {lr:.3e} ' \
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                      epoch,
                      data_iter_step, len(data_loader),
                      100. * data_iter_step / last_idx,
                      loss=losses_m,
                      batch_time=batch_time_m,
                      rate=samples.size(0) * args.world_size / batch_time_m.val,
                      rate_avg=samples.size(0) * args.world_size / batch_time_m.avg,
                      lr=lr,
                      data_time=data_time_m)
                  )
        
        if saver is not None and args.recovery_interval and (
                last_batch or (data_iter_step + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=data_iter_step)
        
        end = time.time()

    return OrderedDict([('loss', losses_m.avg)])

@torch.no_grad()
def evaluate(data_loader, model, device, args, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    # switch to evaluation mode
    model.eval()
    end = time.time()
    last_idx = len(data_loader) - 1
    for batch_idx, (images, target) in enumerate(data_loader):
        last_batch = batch_idx == last_idx

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data, args.world_size)
            acc1 = reduce_tensor(acc1, args.world_size)
            acc5 = reduce_tensor(acc5, args.world_size)
        else:
            reduced_loss = loss.data
        
        torch.cuda.synchronize()
        
        losses_m.update(reduced_loss.item(), images.size(0))
        top1_m.update(acc1.item(), output.size(0))
        top5_m.update(acc5.item(), output.size(0))
            
        batch_time_m.update(time.time() - end)
        end = time.time()
        if (last_batch or batch_idx % args.log_interval == 0):
            log_name = 'Test'
            log_name = 'Test'
            print(f'{log_name}: [{batch_idx:>4d}/{last_idx}] ' \
                  f'Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f}) ' \
                  f'Loss: {losses_m.val:>7.4f} ({losses_m.avg:>6.4f}) ' \
                  f'Acc@1: {top1_m.val:>7.4f} ({top1_m.avg:>7.4f}) ' \
                  f'Acc@5: {top5_m.val:>7.4f} ({top5_m.avg:>7.4f}) '
            )

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics
