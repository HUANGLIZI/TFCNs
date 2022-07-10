import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from preprocess import TFCNs_dataset
from utils import test_single_volume
from networks.TFCNs_model import TFCNs
from networks.TFCNs_configs import get_TFCNs_config
parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,default='/home/lzh/ldh/TFCNs_main/data/test_vol_h5', help='root dir for validation volume data') 
parser.add_argument('--dataset', type=str,default='MMWHS', help='experiment_name')
parser.add_argument('--num_classes', type=int,default=8, help='output channel of network')
parser.add_argument('--list_dir', type=str,default='./lists', help='list dir')
parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=1, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=12,help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.005, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
args = parser.parse_args()


def inference(args, model, test_save_path=None):
    db_test = TFCNs_dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=1)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f mean_jaccard %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1], np.mean(metric_i, axis=0)[2]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f mean_jaccard %f' % (i, metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mean_jaccard = np.mean(metric_list, axis=0)[2]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f mean_jaccard : %f' % (performance, mean_hd95, mean_jaccard))
    return "Testing Finished!"


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

    config_vit = get_TFCNs_config()
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.grid = (int(args.img_size/16), int(args.img_size/16))
    net = TFCNs(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    snapshot = os.path.join('/home/lzh/ldh/TFCNs_main/model', 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    net.load_state_dict(torch.load(snapshot))
    logging.basicConfig(filename="test_log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    test_save_path = None
    inference(args, net, test_save_path)


