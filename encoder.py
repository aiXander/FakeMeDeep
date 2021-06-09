import os
from argparse import Namespace
import time
import os
import sys
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import dlib, cv2
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(".")
sys.path.append("..")
sys.path.append("encoder4editing")

from utils.alignment import align_face
from utils.common import tensor2im
from models.psp import pSp  # we use the pSp framework to load the e4e encoder.
from editings import latent_editor
import config as cfg

def write_video(outpath, image_list, video_fps, rewind = False):
    h, w, c = image_list[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    #fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    
    video = cv2.VideoWriter(outpath, fourcc, video_fps, (w, h))

    for image in image_list:
        video.write(np.uint8(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        
    if rewind:
        for image in image_list[::-1][1:]:
            video.write(np.uint8(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))  

    cv2.destroyAllWindows()
    video.release()


class StyleGAN_Encoder():
    def __init__(self, debug = 0):

        self.verbose = debug

        self.img_transforms = transforms.Compose([
                                transforms.Resize((256, 256)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.resize_dims = (256, 256)
        self.output_size = 1024

        # Load submodules:
        self.net       = self.load_e4e_model(cfg.e4e_model_path)
        self.editor    = latent_editor.LatentEditor(self.net.decoder, False)
        self.predictor = dlib.shape_predictor(cfg.dlib_shape_predictor_path)

    def load_e4e_model(self, model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = model_path
        opts = Namespace(**opts)
        net  = pSp(opts)
        net.eval()
        net.cuda()
        if self.verbose:
            print('e4e model successfully loaded!')
        return net

    def align_one_img(self, img_name, img_path, fix = 1):
        target_folder = os.path.dirname(img_path)

        if fix: #this helps getting rid of potential rotations in the fileheader when taken with eg smartphone
            img = cv2.imread(img_path)
            cv2.imwrite(img_path, img)
            
        try:
            aligned_image = align_face(filepath=img_path, predictor=self.predictor, 
                output_size = self.output_size, transform_size = self.output_size)
        except:
            raise ValueError('No face found in %s' %img_name)

        aligned_HD_img_path = os.path.join(target_folder, Path(img_path).stem + '_aligned_HD.jpg')
        aligned_LD_img_path = os.path.join(target_folder, Path(img_path).stem + '_aligned_LD.jpg')

        aligned_image.save(aligned_HD_img_path)
        aligned_image = aligned_image.resize(self.resize_dims)
        aligned_image.save(aligned_LD_img_path)

        return aligned_HD_img_path, aligned_LD_img_path

    def predict_latents(self, aligned_HD_img_path, aligned_LD_img_path, img_name):
        target_folder = os.path.dirname(aligned_LD_img_path)

        with torch.no_grad():
            input_image_LD       = Image.open(aligned_LD_img_path) 
            input_image_HD       = Image.open(aligned_HD_img_path) 
            
            transformed_image = self.img_transforms(input_image_LD)
            images, latents   = self.net(transformed_image.unsqueeze(0).to("cuda").float(), randomize_noise=False, return_latents=True, resize = False)
            result_image, latent = images[0], latents[0]

            self.save_sbs(tensor2im(result_image), input_image_HD, target_folder, img_name + "_sbs.jpg")
            np.save(os.path.join(target_folder, img_name + '.npy'), latent.cpu().numpy())

        return latent

    def make_interpolation_video(self, latent, direction, direction_name, img_name, latent_range,
        outdir, aligned_HD_img_path):
        up   = np.linspace(0, latent_range[1], num=cfg.n_frames//4)
        down = np.linspace(0, latent_range[0], num=cfg.n_frames//4)
        multipliers = np.concatenate((up, up[::-1], down, down[::-1]))
        
        interpolation_images = self.editor.interfacegan_interpolate(latent, direction, multipliers)
        results = np.stack([np.array(res) for res in interpolation_images])
        
        if cfg.concat_source:
            source_image = np.array(Image.open(aligned_HD_img_path))[np.newaxis, :]
            source_image = np.repeat(source_image, results.shape[0], axis=0)
            results = np.concatenate((source_image, results), axis=cfg.concat_direction)

        video_path = os.path.join(outdir, img_name + '_%s.mp4' %direction_name)
        write_video(video_path, results, cfg.video_out_fps)


    def save_sbs(self, result_image, source_image, output_dir, img_name):
        res = np.concatenate([np.array(source_image),
                              np.array(result_image)], axis=1)
        img = Image.fromarray(res)
        img.save(os.path.join(output_dir, img_name))
        return


def load_face_latents(folder):
    latent_names = [''.join(f.split('.')[:-1]) for f in os.listdir(folder) if '.npy' in f]
    latent_paths = [os.path.join(folder, f) for f in os.listdir(folder) if '.npy' in f]
    latents = []
    for f in latent_paths:
        latent = np.load(f)
        latent = torch.Tensor(latent).to("cuda")
        latents.append(latent)
    return torch.stack(latents), latent_names