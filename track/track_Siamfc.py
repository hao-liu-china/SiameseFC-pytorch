import torch
from torch.autograd import Variable

import cv2
import argparse
import os
import json
import numpy as np
import time as time
from PIL import Image

from net import AlexNet, Siamfc
from eval_otb import eval_auc
from utils import get_template_z, get_search_x

class track_config(object):
    numScale = 3
    scaleStep = 1.0375
    scalePenalty = 0.9745
    scaleLR = 0.59
    responseUp = 16
    wInfluence = 0.176
    z_lr = 0.1
    scale_min = 0.2
    scale_max = 5

    exemplarSize = 127
    instanceSize = 255
    scoreSize = 17
    totalStride = 8
    contextAmount = 0.5
    final_sz = responseUp * (scoreSize-1) + 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Siamfc on OTB')
    parser.add_argument('--dataset', default='OTB2015', choices=['OTB2013', 'OTB2015'], help='test on which dataset')
    parser.add_argument('--model', default='2016-08-17.net.mat', help='path to trained model')
    args = parser.parse_args()

    dataset = args.dataset                                          # OTB2013 or OTB2015
    base_path = os.path.join('dataset', dataset)                   # path to OTB2015/OTB2013 (image path)
    json_path = os.path.join('dataset', dataset + '.json')        # path to OTB2015.json/OTB2013.json (annotations)
    annos = json.load(open(json_path, 'r'))
    videos = sorted(annos.keys())                                   # video name

    use_gpu = True
    visualization = True

    config = track_config()                                         # tracking parameters
    model = Siamfc(branch=AlexNet())                                # Siamfc tracker
#    model.load_param(path=args.model)                              # load trained model parameters from pth file
    model.load_params_from_mat(net_path=args.model)                 # load trained model parameters from mat file
    model.eval().cuda()                                             # switch to evaluate mode

    speed = []
    # loop videos
    for video_id, video in enumerate(videos):
        video_path_name = annos[video]['name']                      # tracked video

        frame_name_list = [os.path.join(base_path, video_path_name, 'img', im_f) for im_f in annos[video]['image_files']]   # path to tracked frames
        frame_name_list.sort()
        with Image.open(frame_name_list[0]) as img:                 # get frame size
            frame_sz = np.asarray(img.size)
            frame_sz[1], frame_sz[0] = frame_sz[0], frame_sz[1]
        gt = np.array(annos[video]['gt_rect']).astype(np.float)     # groundtruth of tracked video
        n_frames = len(frame_name_list)                              # number of tracked frames
        assert n_frames == len(gt)

        init_rect = np.array(annos[video]['init_rect']).astype(np.float)
        pos_x = init_rect[0] + init_rect[2]/2
        pos_y = init_rect[1] + init_rect[3]/2
        target_w = init_rect[2]
        target_h = init_rect[3]

        num_frames = np.size(frame_name_list)                       # the number of frames in tracked video
        bboxes = np.zeros((num_frames, 4))                          # tracking results
        scale_factors = config.scaleStep ** np.linspace(-np.ceil(config.numScale/2), np.ceil(config.numScale/2), config.numScale)
        hann_1d = np.expand_dims(np.hanning(config.final_sz), axis=0)# hanning window
        penalty = np.transpose(hann_1d) * hann_1d
        penalty = penalty / np.sum(penalty)

        context = config.contextAmount * (target_w + target_h)      # padding
        z_sz = np.sqrt(np.prod((target_w + context) * (target_h + context)))    # original template size
        x_sz = float(config.instanceSize) / config.exemplarSize * z_sz          # original search size

        min_z = config.scale_min * z_sz
        max_z = config.scale_max * z_sz
        min_x = config.scale_min * x_sz
        max_x = config.scale_max * x_sz

        bboxes[0, :] = pos_x-target_w/2, pos_y-target_h/2, target_w, target_h
        z_crop = get_template_z(pos_x, pos_y, z_sz, frame_name_list[0], config)                  # template [1, 3, 127, 127]
        z_crops = torch.stack((z_crop, z_crop, z_crop))                                          # [3, 3, 127, 127]
        template_z = model.branch(Variable(z_crops).cuda())                                          # feature of template [1, 256, 6, 6]
        tic = time.time()                                                                       # start

        for f in range(1, num_frames):  # track
            scaled_exemplar = z_sz * scale_factors
            scaled_search_area = x_sz * scale_factors
            scaled_target_w = target_w * scale_factors
            scaled_target_h = target_h * scale_factors

            x_crops = get_search_x(pos_x, pos_y, scaled_search_area, frame_name_list[f], config)# search [3, 3, 255, 255]
            template_x = model.branch(Variable(x_crops).cuda())                                 # [3, 256, 22, 22]
            scores = model.Xcorr(template_x, template_z)                                        # [3, 1, 17, 17]
            scores = model.bn_adjust(scores)                                                    # [3, 1, 17, 17]

            scores = scores.squeeze().permute(1, 2, 0).data.cpu().numpy()                       # [3,1,17,17] -> [3,17,17] -> [17,17,3]
            scores_up = cv2.resize(scores, (config.final_sz, config.final_sz), interpolation=cv2.INTER_CUBIC)   # [257,257,3]
            scores_up = scores_up.transpose((2, 0, 1))                                                          # [3,257,257]

            scores_ = np.squeeze(scores_up)
            # penalize change of scale
            scores_[0, :, :] = config.scalePenalty * scores_[0, :, :]
            scores_[2, :, :] = config.scalePenalty * scores_[2, :, :]
            # find scale with highest peak (after penalty)
            new_scale_id = np.argmax(np.amax(scores_, axis=(1, 2)))
            # update scaled sizes
            x_sz = (1 - config.scaleLR) * x_sz + config.scaleLR * scaled_search_area[new_scale_id]
            target_w = (1 - config.scaleLR) * target_w + config.scaleLR * scaled_target_w[new_scale_id]
            target_h = (1 - config.scaleLR) * target_h + config.scaleLR * scaled_target_h[new_scale_id]

            # select response with new_scale_id
            score_ = scores_[new_scale_id, :, :]
            score_ = score_ - np.min(score_)
            score_ = score_ / np.sum(score_)
            # apply displacement penalty
            score_ = (1 - config.wInfluence) * score_ + config.wInfluence * penalty
            p = np.asarray(np.unravel_index(np.argmax(score_), np.shape(score_)))                   # position of max response in score_
            center = float(config.final_sz - 1) / 2                                                 # center of score_
            disp_in_area = p - center
            disp_in_xcrop = disp_in_area * float(config.totalStride) / config.responseUp
            disp_in_frame = disp_in_xcrop * x_sz / config.instanceSize
            pos_y, pos_x = pos_y + disp_in_frame[0], pos_x + disp_in_frame[1]
            bboxes[f, :] = pos_x - target_w / 2, pos_y - target_h / 2, target_w, target_h

            # update template
            #if config.z_lr > 0:
            #    new_z_crop = get_template_z(pos_x, pos_y, z_sz, frame_name_list[f], config).unsqueeze(0)
                #new_z_crop = model.branch(Variable(new_z_crop).cuda())
            #    z_crop = (1 - config.z_lr) * z_crop + config.z_lr * new_z_crop

            z_sz = (1 - config.scaleLR) * z_sz + config.scaleLR * scaled_exemplar[new_scale_id]

            if visualization:
                im_show = cv2.cvtColor(cv2.imread(frame_name_list[f]), cv2.COLOR_RGB2BGR)
                cv2.rectangle(im_show, (int(pos_x - target_w/2), int(pos_y - target_h/2)), (int(pos_x + target_w/2), int(pos_y + target_h/2)),
                              (0, 255, 0), 3)
                cv2.putText(im_show, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(video, im_show)
                cv2.waitKey(1)

        toc = time.time() - tic
        fps = num_frames / toc
        speed.append(fps)
        print('{:3d} Video: {:12s} Time: {:3.1f}s\tSpeed: {:3.1f}fps'.format(video_id, video, toc, fps))

        # save result
        test_path = os.path.join('result', dataset, 'Siamese-fc_test')
        if not os.path.isdir(test_path): os.makedirs(test_path)
        result_path = os.path.join(test_path, video + '.txt')
        with open(result_path, 'w') as f:
            for x in bboxes:
                f.write(','.join(['{:.2f}'.format(i) for i in x]) + '\n')

    eval_auc(dataset, 'Siamese-fc_test', 0, 1)