# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import math
import pickle

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
from tqdm import tqdm

from models.pose_estimator.lib.config import cfg
from models.pose_estimator.lib.config import update_config
from models.pose_estimator.lib.core.inference import get_max_preds




from models.pose_estimator.lib.utils.utils import create_logger



def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args


def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2,return_image=False):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            #joints_vis = batch_joints_vis[k]

            for joint_id,(joint, joint_vis) in enumerate(zip(joints, joints)):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                # if joint_vis[0]:

                cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
                ndarr = cv2.putText(ndarr,str(joint_id),(int(joint[0])+5, int(joint[1])+5),cv2.FONT_HERSHEY_SIMPLEX,.2,(0,255,0),1)
            k = k + 1

    if return_image:
        return ndarr
    ndarr = cv2.cvtColor(ndarr,cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_name, ndarr)


def save_debug_images(config, input,# meta,
                      joints_pred,
                      prefix):


    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input, joints_pred, [],
            #meta['joints_vis'],
            '{}_pred.jpg'.format(prefix)
        )

def infer(config, val_loader, val_dataset, model, output_dir,device,num_keypoints=16):

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    data_dict = val_dataset.data.copy()
    print(data_dict['flow_paths'][0])
    data_dict.update({'keypoints_abs': np.zeros((len(val_dataset),num_keypoints,2)),
                      'keypoints_rel': np.zeros((len(val_dataset),num_keypoints,2))})
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader,desc=f'Estimating keypoints')):
            # compute output
            input = batch['img'].to(device)
            ids = batch['id'].numpy()
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            num_images = input.size(0)

            # multiply by four,to obtain resolution of 256x256
            pred, _ = get_max_preds(output.cpu().numpy())
            pred *= 4


            idx += num_images
            data_dict['keypoints_abs'][ids] = pred
            keypoints_rel = pred / 256.
            data_dict['keypoints_rel'][ids] = keypoints_rel

            if i % config.PRINT_FREQ == 0:
                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input,
                                  #meta,
                                  pred,
                                  prefix)

        print(data_dict['flow_paths'][0])
        with open(os.path.join(val_dataset.datapath,'meta_kp.p'), 'wb') as f:
            pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    from models.pose_estimator.lib.dataset.infer_datasets import InferenceDataset
    from models.pose_estimator.lib.models.pose_resnet import get_pose_net as get_pose_resnet
    from models.pose_estimator.lib.models.pose_hrnet import get_pose_net as get_pose_hrnet

    model_fns = {'pose_resnet': get_pose_resnet, 'pose_hrnet': get_pose_hrnet}

    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    # fixme not working
    model = model_fns[cfg.MODEL.NAME](
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:

        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE,map_location='cpu'), strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    device = torch.device(f'cuda:{cfg.GPUS}' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    # # define loss function (criterion) and optimizer
    # criterion = JointsMSELoss(
    #     use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    # ).cuda()
    valid_dataset = InferenceDataset(dataset=cfg.DATASET.DATASET)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU,
        shuffle=True,
        num_workers=cfg.WORKERS,
        pin_memory=True,
        drop_last=False
    )

    # evaluate on validation set
    infer(cfg, valid_loader, valid_dataset, model,final_output_dir,device=device)


if __name__ == '__main__':
    main()
