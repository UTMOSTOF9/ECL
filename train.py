#!/usr/bin/env python3
# -*- encodinng: uft-8 -*-
'''
@file: train.py
@author:zyl
@contact:zhangyilan@buaa.edu.cn
@time:2023/7/8 17:38
'''


import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader

from models.ecl import balanced_proxies, build_model
from models.loss import BHP, CE_weight
from utils.dataset.isic import (augmentation_rand, augmentation_sim,
                                augmentation_test, isic2018_dataset,
                                isic2019_dataset)
from utils.eval_metrics import Auc, ConfusionMatrix

'''function for saving model'''

cv2.setNumThreads(0)
torch.set_num_threads(1)


def model_snapshot(model, new_modelpath, old_modelpath=None, only_bestmodel=False):
    if only_bestmodel and old_modelpath:
        os.remove(old_modelpath)
    torch.save(model.state_dict(), new_modelpath)


'''function for getting proxies number'''


def get_proxies_num(cls_num_list):
    ratios = [max(np.array(cls_num_list)) / num for num in cls_num_list]
    prototype_num_list = []
    for ratio in ratios:
        if ratio == 1:
            prototype_num = 1
        else:
            prototype_num = int(ratio // 10) + 2
        prototype_num_list.append(prototype_num)
    assert len(prototype_num_list) == len(cls_num_list)
    return prototype_num_list


def main(args):

    wandb.init(
        project="cvpdl-final",
        name=args.exp_name,
        dir=args.log_path,
        config=vars(args),
    )

    Path(args.log_path).mkdir(parents=True, exist_ok=True)
    log_file = open(os.path.join(args.log_path, 'train_log.txt'), 'w')

    '''print args'''
    for arg in vars(args):
        print(arg, getattr(args, arg))
        print(arg, getattr(args, arg), file=log_file)

    '''load models'''
    model = build_model(name=args.backbone, num_classes=args.num_classes, feat_dim=args.feat_dim)
    proxy_num_list = get_proxies_num(args.cls_num_list)
    model_proxy = balanced_proxies(dim=args.feat_dim, proxy_num=sum(proxy_num_list))

    if args.cuda:
        model.cuda()
        model_proxy.cuda()
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0), file=log_file)
    print("Model_proxy size: {:.5f}M".format(sum(p.numel() for p in model_proxy.parameters()) / 1000000.0))
    print("Model_proxy size: {:.5f}M".format(sum(p.numel()
          for p in model_proxy.parameters()) / 1000000.0), file=log_file)
    print("=============model init done=============")
    print("=============model init done=============", file=log_file)

    complete = False

    '''load dataset'''
    transfrom_train = [augmentation_rand, augmentation_sim]
    if args.dataset == 'ISIC2018':
        train_iterator = DataLoader(
            isic2018_dataset(path=args.data_path, transform=transfrom_train, mode='train'),
            batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True
        )
        valid_iterator = DataLoader(
            isic2018_dataset(path=args.data_path, transform=augmentation_test, mode='valid'),
            batch_size=1, shuffle=False, num_workers=4
        )
        test_iterator = DataLoader(
            isic2018_dataset(path=args.data_path, transform=augmentation_test, mode='test'),
            batch_size=1, shuffle=False, num_workers=4
        )

    elif args.dataset == 'ISIC2019':
        train_iterator = DataLoader(
            isic2019_dataset(path=args.data_path, transform=transfrom_train, mode='train'),
            batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True
        )
        valid_iterator = DataLoader(
            isic2019_dataset(path=args.data_path, transform=augmentation_test, mode='valid'),
            batch_size=1, shuffle=False, num_workers=4
        )
        test_iterator = DataLoader(
            isic2019_dataset(path=args.data_path, transform=augmentation_test, mode='test'),
            batch_size=1, shuffle=False, num_workers=4
        )
    else:
        raise ValueError("dataset error")

    '''load optimizer'''

    # parameters = [p for p in model.backbone.parameters()] + [p for p in model.ssl_branch.parameters()]
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)

    # parameters = [p for p in model.backbone.parameters()] + [p for p in model.ssl_branch.parameters()]
    # optimizer1 = optim.SGD(parameters, lr=0.002, weight_decay=0.0001, momentum=0.9)

    # parameters = [p for p in model.clip_branch.parameters()] + [p for p in model.classifier.parameters()]
    # optimizer2 = optim.SGD(parameters, lr=0.002, weight_decay=0.01, momentum=0.9)

    optimizer_proxies = optim.SGD(model_proxy.parameters(), lr=0.002, weight_decay=0.0001, momentum=0.9)

    # cosine lr
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_iterator))
    # lr_scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=args.epochs * len(train_iterator))
    # lr_scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=args.epochs * len(train_iterator))
    lr_scheduler_proxies = optim.lr_scheduler.CosineAnnealingLR(optimizer_proxies, T_max=args.epochs)

    '''load loss'''
    criterion_ce = CE_weight(cls_num_list=args.cls_num_list, E1=args.E1, E2=args.E2, E=args.epochs)
    criterion_bhp = BHP(cls_num_list=args.cls_num_list, proxy_num_list=proxy_num_list)
    criterion_map = nn.SmoothL1Loss(beta=0.01)
    alpha = args.alpha
    beta = args.beta

    '''train'''
    f_score_list = [1.0 for _ in range(args.num_classes)]
    steps = 0
    best_acc = 0.0
    old_model_path = None
    curr_patience = args.patience
    start_time = time.time()
    try:
        for e in range(args.epochs):
            model.train()
            model_proxy.train()
            print('Epoch: {}'.format(e))
            print('Epoch: {}'.format(e), file=log_file)

            start_time_epoch = time.time()
            train_loss = 0.0

            optimizer_proxies.zero_grad()

            for batch_index, (data, label) in enumerate(train_iterator):
                if args.cuda:
                    for i in range(len(data)):
                        data[i] = data[i].cuda()
                    label = label.cuda()
                diagnosis_label = label.squeeze(1)

                optimizer.zero_grad()
                # optimizer1.zero_grad()
                # optimizer2.zero_grad()

                with torch.cuda.amp.autocast(dtype=torch.bfloat16 if args.bf16 else torch.float32):
                    output, feat_mlp, reconstruct_maps = model(data)
                    output_proxy = model_proxy()
                    feat_mlp = torch.cat([feat_mlp[0].unsqueeze(1), feat_mlp[1].unsqueeze(1)], dim=1)

                    loss_ce = criterion_ce(output, diagnosis_label, (e + 1), f_score_list)
                    loss_bhp = criterion_bhp(output_proxy, feat_mlp, diagnosis_label)

                    with torch.no_grad():
                        reconstruct_targets1 = interpolate(data[0], size=reconstruct_maps[0].shape[2:])
                        reconstruct_targets2 = interpolate(data[1], size=reconstruct_maps[0].shape[2:])

                    loss_map = criterion_map(reconstruct_maps[0], reconstruct_targets1)
                    loss_map += criterion_map(reconstruct_maps[1], reconstruct_targets2)
                    loss = alpha * loss_ce + beta * loss_bhp + 0.2 * loss_map

                    wandb.log(
                        {
                            "loss": loss,
                            "loss_ce": loss_ce,
                            "loss_bhp": loss_bhp,
                            "loss_map": loss_map
                        },
                        step=steps
                    )
                    steps += 1

                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                # optimizer1.step()
                # optimizer2.step()
                # lr_scheduler1.step()
                # lr_scheduler2.step()

                train_loss += loss.item()

                if batch_index % 50 == 0 and batch_index != 0:
                    predicted_results = torch.argmax(output, dim=1)
                    correct_num = (predicted_results.cpu() == diagnosis_label.cpu()).sum().item()
                    acc = correct_num / len(diagnosis_label)
                    train_log = (
                        f'Training epoch: {e} [{batch_index * args.batch_size}/{len(train_iterator.dataset)}], '
                        f'Loss: {loss.item():.4f}, Accuracy: {acc:.4f}, '
                        f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}'
                        # f'Learning rate 1: {optimizer1.param_groups[0]["lr"]:.6f}, '
                        # f'Learning rate 2: {optimizer2.param_groups[0]["lr"]:.6f}'
                    )
                    wandb.log(
                        {
                            "lr": optimizer.param_groups[0]["lr"],
                            # "lr_1": optimizer1.param_groups[0]["lr"],
                            # 'lr_2': optimizer2.param_groups[0]['lr']
                        },
                        step=steps,
                    )
                    print(train_log)
                    print(train_log, file=log_file)

            optimizer_proxies.step()
            lr_scheduler_proxies.step()

            train_epoch_log = f"Epoch {e} complete! Average Training loss: {train_loss / len(train_iterator):.4f}"
            print(train_epoch_log)
            print(train_epoch_log, file=log_file)

            # Validation
            model.eval()
            model_proxy.eval()
            pro_diag, lab_diag = [], []
            val_confusion_diag = ConfusionMatrix(num_classes=args.num_classes, labels=list(range(args.num_classes)))
            valid_loss_ce = 0.0
            with torch.no_grad():
                for batch_index, (data, label) in enumerate(valid_iterator):
                    if args.cuda:
                        data = data.cuda()
                        label = label.cuda()
                    diagnosis_label = label.squeeze(1)

                    with torch.cuda.amp.autocast(dtype=torch.float32):
                        output = model(data)
                        output = output.float()
                        predicted_results = torch.argmax(output, dim=1)
                        pro_diag.extend(output.detach().cpu().numpy())
                        lab_diag.extend(diagnosis_label.cpu().numpy())
                        val_confusion_diag.update(predicted_results.cpu().numpy(), diagnosis_label.cpu().numpy())
                        valid_loss_ce += criterion_ce(output, diagnosis_label, (e + 1),
                                                      f_score_list) / len(valid_iterator)

                dia_acc = val_confusion_diag.summary(log_file)
                mean_metrics = val_confusion_diag.get_metrics()
                roc_auc = Auc(pro_diag, lab_diag, args.num_classes, log_file)
                wandb.log(
                    {
                        "valid_loss_ce": valid_loss_ce,
                        "val_acc": dia_acc,
                        "val_auc": np.mean(roc_auc),
                        'val_precision': mean_metrics['precision'],
                        'val_sensitivity': mean_metrics['sensitivity'],
                        'val_specificity': mean_metrics['specificity'],
                        'val_f1_score': mean_metrics['f1_score'],
                    },
                    step=steps
                )
                f_score_list = val_confusion_diag.get_f1score()

                end_time_epoch = time.time()
                training_time_epoch = end_time_epoch - start_time_epoch
                total_training_time = time.time() - start_time
                remaining_time = training_time_epoch * args.epochs - total_training_time
                val_log = (
                    f"Total training time: {total_training_time:.4f}s, "
                    f"{training_time_epoch:.4f} s/epoch, "
                    f"Estimated remaining time: {remaining_time:.4f}s, "
                    f"Val metrics: {mean_metrics}, \n"
                    f"Feat_Fusion_weight: {model.fusion_weight}, \n"
                )
                print(val_log)
                print(val_log, file=log_file)

                if dia_acc > best_acc:
                    curr_patience = args.patience
                    best_acc = dia_acc
                    new_model_path = os.path.join(args.model_path, 'bestacc_model_{}.pth'.format(e))
                    model_snapshot(model, new_model_path, old_modelpath=old_model_path, only_bestmodel=True)
                    old_model_path = new_model_path
                    print("Found new best model, saving to disk...")
                else:
                    curr_patience -= 1
                    if curr_patience == 0:
                        print("Early stopping, best accuracy: {:.4f}".format(best_acc))
                        print("Early stopping, best accuracy: {:.4f}".format(best_acc), file=log_file)
                        complete = True
                        break

                if e == args.epochs - 1:
                    print("Training complete, best accuracy: {:.4f}".format(best_acc))
                    print("Training complete, best accuracy: {:.4f}".format(best_acc), file=log_file)
                    complete = True
            log_file.flush()

        # Test
        if complete:
            model.load_state_dict(torch.load(old_model_path), strict=True)
            model.eval()

            pro_diag, lab_diag = [], []
            confusion_diag = ConfusionMatrix(num_classes=args.num_classes, labels=list(range(args.num_classes)))
            with torch.no_grad():
                for batch_index, (data, label) in enumerate(test_iterator):
                    if args.cuda:
                        data = data.cuda()
                        label = label.cuda()
                    diagnosis_label = label.squeeze(1)

                    output = model(data)
                    predicted_results = torch.argmax(output, dim=1)
                    pro_diag.extend(output.detach().cpu().numpy())
                    lab_diag.extend(diagnosis_label.cpu().numpy())

                    confusion_diag.update(predicted_results.cpu().numpy(), diagnosis_label.cpu().numpy())

                print("Test confusion matrix:")
                print("Test confusion matrix:", file=log_file)
                test_acc = confusion_diag.summary(log_file)
                mean_metrics = confusion_diag.get_metrics()
                print("Test AUC:")
                print("Test AUC:", file=log_file)
                roc_auc = Auc(pro_diag, lab_diag, args.num_classes, log_file)

                print(f"Test metrics: {mean_metrics}")
                print(f"Test metrics: {mean_metrics}", file=log_file)
                wandb.log(
                    {
                        "test_acc": test_acc,
                        "test_auc": np.mean(roc_auc),
                        'test_precision': mean_metrics['precision'],
                        'test_sensitivity': mean_metrics['sensitivity'],
                        'test_specificity': mean_metrics['specificity'],
                        'test_f1_score': mean_metrics['f1_score'],
                    },
                    step=steps
                )

    except Exception:
        import traceback
        traceback.print_exc()

    finally:
        log_file.close()


def _seed_torch(args):
    r"""
    Sets custom seed for torch

    Args:
        - seed : Int

    Returns:
        - None

    """
    import random
    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        else:
            raise EnvironmentError("GPU device not found")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser(description='Training for the classification task')

# dataset
parser.add_argument('--data_path', type=str, default='/media/disk/zyl/data/ISIC_CL/ISIC2018/',
                    help='the path of the data')
parser.add_argument('--dataset', type=str, default='ISIC2018',
                    choices=['ISIC2018', 'ISIC2019'], help='the name of the dataset')
parser.add_argument('--model_path', type=str,
                    default="/media/disk/zyl/Experiment/ISIC_CL/ISIC2018/test_git/", help='the path of the model')
parser.add_argument('--log_path', type=str, default=None, help='the path of the log')


# training parameters
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
parser.add_argument('--cuda', type=bool, default=True, help='whether to use cuda')
parser.add_argument('--bf16', type=bool, default=False, help='whether to use cuda')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--backbone', type=str, default='ResNet50', help='model name')
parser.add_argument('--exp_name', type=str, default='', help='exp name')


# loss weights
parser.add_argument('--alpha', type=float, default=2.0, choices=[0.25, 0.5, 1.0, 2.0], help='weight of the CE loss')
parser.add_argument('--beta', type=float, default=1.0, choices=[0.25, 0.5, 1.0, 2.0], help='weight of the BHP loss')
# hyperparameters for ce loss
parser.add_argument('--E1', type=int, default=20, choices=[20, 30, 40], help='hyperparameter for ce loss')
parser.add_argument('--E2', type=int, default=50, choices=[50, 60, 70], help='hyperparameter for ce loss')

# hyperparameters for model
parser.add_argument('--feat_dim', dest='feat_dim', type=int, default=128)


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    _seed_torch(args)
    if args.dataset == 'ISIC2018':
        args.cls_num_list = [84, 195, 69, 4023, 308, 659, 667]
        args.num_classes = 7
    elif args.dataset == 'ISIC2019':
        args.cls_num_list = [519, 1993, 1574, 143, 2712, 7725, 376, 151]
        args.num_classes = 8
    else:
        raise Exception("Invalid dataset name!")

    if args.log_path is None:
        args.log_path = args.model_path
    main(args)
    print("Done!")
