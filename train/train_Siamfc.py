import os
import shutil
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from os.path import isfile, join, isdir

from dataset import Pair
from net import AlexNet, Siamfc
from loss import BCEWeightLoss
from utils import adjust_learning_rate, AverageMeter, get_template_z, get_search_x

import json
from PIL import Image
import numpy as np
import time as time
import cv2
from eval_otb import eval_auc


parser = argparse.ArgumentParser(description='Training Siamfc in Pytorch 0.4.0')
parser.add_argument('--numEpochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--lr', default=0.0001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('--weight-decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--batch-size', default=8, type=int, metavar='N', help='mini-batch size (default: 48)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--save', default='./model', type=str, help='directory for saving')
parser.add_argument('--dataset', default='/home/pylab/LHWorkspace/ILSVRC2015', type=str, help='path to original ILSVRC2015-VID')
parser.add_argument('--stats-path', default='ILSVRC2015.stats.mat', type=str, help='path to ILSVRC2015.stats.mat')
args = parser.parse_args()
print(args)

best_auc = 0

# parameters configuration(dataset)
class train_config(object):
    exemplarSize = 127
    instanceSize = 255
    scoreSize = 17
    context = 0.5
    rPos = 16
    rNeg = 0
    totalStride = 8
    ignoreLabel = -100


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


model = Siamfc(branch=AlexNet())
# print(model)                              # print the structure of Siamfc
model.cuda()
gpu_num = torch.cuda.device_count()         # get the number of gpu in current computer
print('GPU NUM: {:2d}'.format(gpu_num))
# multi-gpu train
if gpu_num > 1:
    model = torch.nn.DataParallel(model, device_ids=list(range(gpu_num))).cuda()

criterion = BCEWeightLoss()                 # define criterion

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) # define optimizer

# resume train from a specified epoch
if args.resume:
    if isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_auc = checkpoint['best_auc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

# dataset(train, val)
config = train_config()

transforms_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
transforms_val = transforms.ToTensor()

pair_train = Pair(root_dir=args.dataset, subset='train', transform=transforms_train, config=config, pairs_per_video=40)
pair_val = Pair(root_dir=args.dataset, subset='val', transform=transforms_val, config=config, pairs_per_video=25)

train_loader = torch.utils.data.DataLoader(pair_train, batch_size=args.batch_size*gpu_num, shuffle=True,
                                           num_workers=args.workers, pin_memory=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(pair_val, batch_size=args.batch_size*gpu_num, shuffle=True,
                                         num_workers=args.workers, pin_memory=True, drop_last=True)

# save model
if not isdir(args.save):
    os.makedirs(args.save)

def save_checkpoint(state, epoch):
    filename = join(args.save, '{}_checkpoint.pth.tar'.format(epoch + 1))
    torch.save(state, filename)


# train model
def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()

    # switch to train mode
    model.train()

    # train process
    for idx, (template, search, labels, weights) in enumerate(train_loader):
        template = Variable(template).cuda()
        search = Variable(search).cuda()
        labels = Variable(labels).cuda()
        weights = Variable(weights).cuda()
        # compute output
        output = model(search, template)
        loss = criterion(output, labels, weights)/template.size(0)
        # record loss
        losses.update(loss.item())
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, idx, len(train_loader),loss=losses))
    # save train model
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_auc': best_auc,
        'optimizer': optimizer.state_dict(),
    }, epoch)


# validate model
def validate(val_loader, model, criterion):
    losses = AverageMeter()

    # switch to validate mode
    model.eval()

    # validate process
    for idx, (template, search, labels, weights) in enumerate(val_loader):
        template = Variable(template).cuda()
        search = Variable(search).cuda()
        labels = Variable(labels).cuda()
        weights = Variable(weights).cuda()
        # compute output
        output = model(search, template)
        loss = criterion(output, labels, weights)/(args.batch_size * gpu_num)
        # record loss
        losses.update(loss.item())

        if idx % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                idx, len(val_loader), loss=losses))
    print(' * Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses))

    return losses.avg


def test_otb(epoch):
    model_path = os.path.join(args.save, '{}_checkpoint.pth.tar'.format(epoch + 1))
    #model_path = '2016-08-17.net.mat'
    dataset = 'OTB2015'

    base_path = join('dataset', dataset)
    json_path = join('dataset', dataset + '.json')
    annos = json.load(open(json_path, 'r'))
    videos = sorted(annos.keys())

    use_gpu = True
    visualization = False

    test_config = track_config()  # tracking parameters
    model = Siamfc(branch=AlexNet())  # Siamfc tracker
    #model.load_params_from_mat(net_path=model_path)
    model.load_params(net_path=model_path)
    model.eval().cuda()

    speed = []
    # loop videos
    for video_id, video in enumerate(videos):
        video_path_name = annos[video]['name']  # tracked video

        frame_name_list = [os.path.join(base_path, video_path_name, 'img', im_f) for im_f in annos[video]['image_files']]  # path to tracked frames
        frame_name_list.sort()
        with Image.open(frame_name_list[0]) as img:                 # get frame size
            frame_sz = np.asarray(img.size)
            frame_sz[1], frame_sz[0] = frame_sz[0], frame_sz[1]
        gt = np.array(annos[video]['gt_rect']).astype(np.float)     # groundtruth of tracked video
        n_frames = len(frame_name_list)                              # number of tracked frames
        assert n_frames == len(gt)

        init_rect = np.array(annos[video]['init_rect']).astype(np.float)
        pos_x = init_rect[0] + init_rect[2] / 2
        pos_y = init_rect[1] + init_rect[3] / 2
        target_w = init_rect[2]
        target_h = init_rect[3]

        num_frames = np.size(frame_name_list)  # the number of frames in tracked video
        bboxes = np.zeros((num_frames, 4))  # tracking results
        scale_factors = test_config.scaleStep ** np.linspace(-np.ceil(test_config.numScale / 2), np.ceil(test_config.numScale / 2),
                                                             test_config.numScale)
        hann_1d = np.expand_dims(np.hanning(test_config.final_sz), axis=0)  # hanning window
        penalty = np.transpose(hann_1d) * hann_1d
        penalty = penalty / np.sum(penalty)

        context = test_config.contextAmount * (target_w + target_h)  # padding
        z_sz = np.sqrt(np.prod((target_w + context) * (target_h + context)))  # original template size
        x_sz = float(test_config.instanceSize) / test_config.exemplarSize * z_sz  # original search size

        bboxes[0, :] = pos_x - target_w / 2, pos_y - target_h / 2, target_w, target_h
        z_crop = get_template_z(pos_x, pos_y, z_sz, frame_name_list[0], test_config)  # template [1, 3, 127, 127]
        z_crops = torch.stack((z_crop, z_crop, z_crop))  # [3, 3, 127, 127]
        template_z = model.branch(Variable(z_crops).cuda())  # feature of template [1, 256, 6, 6]
        tic = time.time()

        for f in range(1, num_frames):  # track
            scaled_exemplar = z_sz * scale_factors
            scaled_search_area = x_sz * scale_factors
            scaled_target_w = target_w * scale_factors
            scaled_target_h = target_h * scale_factors

            x_crops = get_search_x(pos_x, pos_y, scaled_search_area, frame_name_list[f], test_config)# search [3, 3, 255, 255]
            template_x = model.branch(Variable(x_crops).cuda())                                 # [3, 256, 22, 22]
            scores = model.Xcorr(template_x, template_z)                                        # [3, 1, 17, 17]
            scores = model.bn_adjust(scores)                                                    # [3, 1, 17, 17]

            scores = scores.squeeze().permute(1, 2, 0).data.cpu().numpy()  # [3,1,17,17] -> [3,17,17] -> [17,17,3]
            scores_up = cv2.resize(scores, (test_config.final_sz, test_config.final_sz),
                                   interpolation=cv2.INTER_CUBIC)  # [257,257,3]
            scores_up = scores_up.transpose((2, 0, 1))

            scores_ = np.squeeze(scores_up)
            # penalize change of scale
            scores_[0, :, :] = test_config.scalePenalty * scores_[0, :, :]
            scores_[2, :, :] = test_config.scalePenalty * scores_[2, :, :]
            # find scale with highest peak (after penalty)
            new_scale_id = np.argmax(np.amax(scores_, axis=(1, 2)))
            # update scaled sizes
            x_sz = (1 - test_config.scaleLR) * x_sz + test_config.scaleLR * scaled_search_area[new_scale_id]
            target_w = (1 - test_config.scaleLR) * target_w + test_config.scaleLR * scaled_target_w[new_scale_id]
            target_h = (1 - test_config.scaleLR) * target_h + test_config.scaleLR * scaled_target_h[new_scale_id]

            # select response with new_scale_id
            score_ = scores_[new_scale_id, :, :]
            score_ = score_ - np.min(score_)
            score_ = score_ / np.sum(score_)
            # apply displacement penalty
            score_ = (1 - test_config.wInfluence) * score_ + test_config.wInfluence * penalty
            p = np.asarray(np.unravel_index(np.argmax(score_), np.shape(score_)))  # position of max response in score_
            center = float(test_config.final_sz - 1) / 2  # center of score_
            disp_in_area = p - center
            disp_in_xcrop = disp_in_area * float(test_config.totalStride) / test_config.responseUp
            disp_in_frame = disp_in_xcrop * x_sz / test_config.instanceSize
            pos_y, pos_x = pos_y + disp_in_frame[0], pos_x + disp_in_frame[1]
            bboxes[f, :] = pos_x - target_w / 2, pos_y - target_h / 2, target_w, target_h

            z_sz = (1 - test_config.scaleLR) * z_sz + test_config.scaleLR * scaled_exemplar[new_scale_id]

            if visualization:
                im_show = cv2.cvtColor(cv2.imread(frame_name_list[f]), cv2.COLOR_RGB2BGR)
                cv2.rectangle(im_show, (int(pos_x - target_w / 2), int(pos_y - target_h / 2)),
                              (int(pos_x + target_w / 2), int(pos_y + target_h / 2)),
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

    auc = eval_auc(dataset, 'Siamese-fc_test', 0, 1)
    return auc

def tune_otb(model_path):
    # model_path = '2016-08-17.net.mat'
    dataset = 'OTB2013'

    base_path = join('dataset', dataset)
    json_path = join('dataset', dataset + '.json')
    annos = json.load(open(json_path, 'r'))
    videos = sorted(annos.keys())

    use_gpu = True
    visualization = False

    test_config = track_config()  # tracking parameters
    model = Siamfc(branch=AlexNet())  # Siamfc tracker
    # model.load_params_from_mat(net_path=model_path)
    model.load_params(net_path=model_path)
    model.eval().cuda()
    for test_config.scaleStep in np.arange(1.01, 1.05, 0.0025, np.float):
        for test_config.scalePenalty in np.arange(0.96, 0.99, 0.0025, np.float):
            #for test_config.scaleLR in np.arange(0.57, 0.61, 0.005, np.float):
            #    for test_config.wInfluence in np.arange(0.17, 0.186, 0.002, np.float):
            print('scaleStep: {}, scalePenalty: {}, scaleLR: {}, wInfluence: {}'.format(test_config.scaleStep, test_config.scalePenalty,
                                                                                                test_config.scaleLR, test_config.wInfluence))
            speed = []
            # loop videos
            for video_id, video in enumerate(videos):
                video_path_name = annos[video]['name']  # tracked video

                frame_name_list = [os.path.join(base_path, video_path_name, 'img', im_f) for im_f in
                                   annos[video]['image_files']]  # path to tracked frames
                frame_name_list.sort()
                with Image.open(frame_name_list[0]) as img:  # get frame size
                    frame_sz = np.asarray(img.size)
                    frame_sz[1], frame_sz[0] = frame_sz[0], frame_sz[1]
                gt = np.array(annos[video]['gt_rect']).astype(np.float)  # groundtruth of tracked video
                n_frames = len(frame_name_list)  # number of tracked frames
                assert n_frames == len(gt)

                init_rect = np.array(annos[video]['init_rect']).astype(np.float)
                pos_x = init_rect[0] + init_rect[2] / 2
                pos_y = init_rect[1] + init_rect[3] / 2
                target_w = init_rect[2]
                target_h = init_rect[3]

                num_frames = np.size(frame_name_list)  # the number of frames in tracked video
                bboxes = np.zeros((num_frames, 4))  # tracking results
                scale_factors = test_config.scaleStep ** np.linspace(-np.ceil(test_config.numScale / 2),
                                                                     np.ceil(test_config.numScale / 2),
                                                                     test_config.numScale)
                hann_1d = np.expand_dims(np.hanning(test_config.final_sz), axis=0)  # hanning window
                penalty = np.transpose(hann_1d) * hann_1d
                penalty = penalty / np.sum(penalty)

                context = test_config.contextAmount * (target_w + target_h)  # padding
                z_sz = np.sqrt(np.prod((target_w + context) * (target_h + context)))  # original template size
                x_sz = float(test_config.instanceSize) / test_config.exemplarSize * z_sz  # original search size

                bboxes[0, :] = pos_x - target_w / 2, pos_y - target_h / 2, target_w, target_h
                z_crop = get_template_z(pos_x, pos_y, z_sz, frame_name_list[0],
                                        test_config)  # template [1, 3, 127, 127]
                z_crops = torch.stack((z_crop, z_crop, z_crop))  # [3, 3, 127, 127]
                template_z = model.branch(Variable(z_crops).cuda())  # feature of template [1, 256, 6, 6]
                tic = time.time()

                for f in range(1, num_frames):  # track
                    scaled_exemplar = z_sz * scale_factors
                    scaled_search_area = x_sz * scale_factors
                    scaled_target_w = target_w * scale_factors
                    scaled_target_h = target_h * scale_factors

                    x_crops = get_search_x(pos_x, pos_y, scaled_search_area, frame_name_list[f],
                                           test_config)  # search [3, 3, 255, 255]
                    template_x = model.branch(Variable(x_crops).cuda())  # [3, 256, 22, 22]
                    scores = model.Xcorr(template_x, template_z)  # [3, 1, 17, 17]
                    scores = model.bn_adjust(scores)  # [3, 1, 17, 17]

                    scores = scores.squeeze().permute(1, 2,
                                                      0).data.cpu().numpy()  # [3,1,17,17] -> [3,17,17] -> [17,17,3]
                    scores_up = cv2.resize(scores, (test_config.final_sz, test_config.final_sz),
                                           interpolation=cv2.INTER_CUBIC)  # [257,257,3]
                    scores_up = scores_up.transpose((2, 0, 1))

                    scores_ = np.squeeze(scores_up)
                    # penalize change of scale
                    scores_[0, :, :] = test_config.scalePenalty * scores_[0, :, :]
                    scores_[2, :, :] = test_config.scalePenalty * scores_[2, :, :]
                    # find scale with highest peak (after penalty)
                    new_scale_id = np.argmax(np.amax(scores_, axis=(1, 2)))
                    # update scaled sizes
                    x_sz = (1 - test_config.scaleLR) * x_sz + test_config.scaleLR * scaled_search_area[new_scale_id]
                    target_w = (1 - test_config.scaleLR) * target_w + test_config.scaleLR * scaled_target_w[
                        new_scale_id]
                    target_h = (1 - test_config.scaleLR) * target_h + test_config.scaleLR * scaled_target_h[
                        new_scale_id]

                    # select response with new_scale_id
                    score_ = scores_[new_scale_id, :, :]
                    score_ = score_ - np.min(score_)
                    score_ = score_ / np.sum(score_)
                    # apply displacement penalty
                    score_ = (1 - test_config.wInfluence) * score_ + test_config.wInfluence * penalty
                    p = np.asarray(
                        np.unravel_index(np.argmax(score_), np.shape(score_)))  # position of max response in score_
                    center = float(test_config.final_sz - 1) / 2  # center of score_
                    disp_in_area = p - center
                    disp_in_xcrop = disp_in_area * float(test_config.totalStride) / test_config.responseUp
                    disp_in_frame = disp_in_xcrop * x_sz / test_config.instanceSize
                    pos_y, pos_x = pos_y + disp_in_frame[0], pos_x + disp_in_frame[1]
                    bboxes[f, :] = pos_x - target_w / 2, pos_y - target_h / 2, target_w, target_h

                    z_sz = (1 - test_config.scaleLR) * z_sz + test_config.scaleLR * scaled_exemplar[new_scale_id]

                    if visualization:
                        im_show = cv2.cvtColor(cv2.imread(frame_name_list[f]), cv2.COLOR_RGB2BGR)
                        cv2.rectangle(im_show, (int(pos_x - target_w / 2), int(pos_y - target_h / 2)),
                                      (int(pos_x + target_w / 2), int(pos_y + target_h / 2)),
                                      (0, 255, 0), 3)
                        cv2.putText(im_show, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2,
                                    cv2.LINE_AA)
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



tune_otb('27_checkpoint.pth.tar')
#for epoch in range(args.start_epoch, args.numEpochs):
    #adjust_learning_rate(optimizer, epoch, args)

    #train(train_loader, model, criterion, optimizer, epoch)

    #loss = validate(val_loader, model, criterion)

    #auc = test_otb(epoch)
    #is_best = auc > best_auc
    #if is_best:
    #    shutil.copyfile(join(args.save, '{}_checkpoint.pth.tar'.format(epoch + 1)), join(args.save, 'model_best.pth.tar'))
    #best_auc = max(auc, best_auc)
    #is_best = loss < best_loss
   # best_loss = min(best_loss, loss)

    #save_checkpoint({
    #        'epoch': epoch + 1,
    #        'state_dict': model.state_dict(),
    #        'best_loss': best_loss,
    #        'optimizer': optimizer.state_dict(),
    #    }, epoch, is_best)