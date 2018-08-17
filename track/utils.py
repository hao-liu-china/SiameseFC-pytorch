import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageStat, ImageOps


# if necessary, you need pad frame with avgChans/0
def pad_frame(im, frame_sz, pos_x, pos_y, patch_sz, avg_chan):
    c = patch_sz / 2
    xleft_pad = max(0, -int(round(pos_x - c)))
    ytop_pad = max(0, -int(round(pos_y - c)))
    xright_pad = max(0, int(round(pos_x + c)) - frame_sz[1])
    ybottom_pad = max(0, int(round(pos_y + c)) - frame_sz[0])
    npad = max((xleft_pad, ytop_pad, xright_pad, ybottom_pad))
    if avg_chan is not None:
        # TODO: PIL Image doesn't allow float RGB image
        avg_chan = tuple([int(round(c)) for c in avg_chan])
        im_padded = ImageOps.expand(im, border=npad, fill=avg_chan)
    else:
        im_padded = ImageOps.expand(im, border=npad, fill=0)
    return im_padded, npad                                       # return padded frame and npad


# extract z_crop (Size: 127) as template
def get_template_z(pos_x, pos_y, z_sz, image, config):
    image = Image.open(image)                           # open image
    if image.mode == 'L':                               # if im is gray image, convert it to RGB
        image = image.convert('RGB')
    avg_chan = ImageStat.Stat(image).mean               # compute mean of three channels
    frame_padded_z, npad_z = pad_frame(image, image.size, pos_x, pos_y, z_sz, avg_chan) # if necessary, pad frame

    c = z_sz / 2
    tr_x = npad_z + int(round(pos_x - c))       # compute x coordinate of top-left corner
    tr_y = npad_z + int(round(pos_y - c))       # compute y coordinate of top-left corner
    width = round(pos_x + c) - round(pos_x - c)
    height = round(pos_y + c) - round(pos_y - c)
    z_crop = frame_padded_z.crop((int(tr_x),
                              int(tr_y),
                              int(tr_x + width),
                              int(tr_y + height)))
    z_crop = z_crop.resize((config.exemplarSize, config.exemplarSize), Image.BILINEAR)
    transform = transforms.ToTensor()
    z_crop = 255.0 * transform(z_crop)
    return z_crop


# extract x_crop (Size: 255) as search
def get_search_x(pos_x, pos_y, scaled_search_area, image, config):
    image = Image.open(image)          # open image
    if image.mode == 'L':                               # if im is gray image, convert it to RGB
        image = image.convert('RGB')
    avg_chan = ImageStat.Stat(image).mean   # compute mean of three channels
    frame_padded_x, npad_x = pad_frame(image, image.size, pos_x, pos_y, scaled_search_area[2], avg_chan)

    # scaledInstance[2] correspondsto the maximum size of image patch
    c = scaled_search_area[2] / 2
    tr_x = npad_x + int(round(pos_x - c))               # compute x coordinate of top-left corner
    tr_y = npad_x + int(round(pos_y - c))               # compute y coordinate of top-left corner
    width = round(pos_x + c) - round(pos_x - c)
    height = round(pos_y + c) - round(pos_y - c)
    search_area = frame_padded_x.crop((int(tr_x),                      # search_area corresponds to scaledInstance[2]
                                       int(tr_y),
                                       int(tr_x + width),
                                       int(tr_y + height)))
    offset_s0 = (scaled_search_area[2] - scaled_search_area[0]) / 2
    offset_s1 = (scaled_search_area[2] - scaled_search_area[1]) / 2

    crop_s0 = search_area.crop((int(offset_s0),                 # crop_x0 corresponds to scaleInstance[0]
                                int(offset_s0),
                                int(offset_s0 + scaled_search_area[0]),
                                int(offset_s0 + scaled_search_area[0])))
    crop_s0 = crop_s0.resize((config.instanceSize, config.instanceSize), Image.BILINEAR)    # x0_crop resize to 255

    crop_s1 = search_area.crop((int(offset_s1),                 # crop_x1 corresponds to scaleInstance[1]
                                int(offset_s1),
                                int(offset_s1 + scaled_search_area[1]),
                                int(offset_s1 + scaled_search_area[1])))
    crop_s1 = crop_s1.resize((config.instanceSize, config.instanceSize), Image.BILINEAR)    # x1_crop resize to 255

    crop_s2 = search_area.resize((config.instanceSize, config.instanceSize), Image.BILINEAR)# x2_crop resize to 255

    transfrom = transforms.ToTensor()
    crop_s0 = 255.0 * transfrom(crop_s0)
    crop_s1 = 255.0 * transfrom(crop_s1)
    crop_s2 = 255.0 * transfrom(crop_s2)
    crops = torch.stack((crop_s0, crop_s1, crop_s2))
    return crops



