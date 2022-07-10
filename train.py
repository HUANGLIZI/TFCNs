import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.TFCNs_model import TFCNs
from networks.TFCNs_configs import get_TFCNs_config
from train_utils import train_starter

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/lzh/ldh/TFCNs_main/data/train_npz', help='dir for training data')
parser.add_argument('--dataset', type=str,
                    default='MMWHS', help='dataset_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=8, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=1,help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=12, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.005,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
args = parser.parse_args()


if __name__ == "__main__":   
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    save_path = "/home/lzh/ldh/TFCNs_main/model"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    config_TFCNs = get_TFCNs_config()
    config_TFCNs.n_classes = args.num_classes
    config_TFCNs.n_skip = args.n_skip
    config_TFCNs.patches.grid = (int(args.img_size / 16), int(args.img_size / 16))
    net = TFCNs(config_TFCNs, img_size=args.img_size, num_classes=config_TFCNs.n_classes).cuda()

    train_starter(args, net, save_path)
