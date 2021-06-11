import os, time, sys, random
from argparse import Namespace
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import dlib, cv2
import matplotlib.pyplot as plt
from pathlib import Path
import scipy

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

def load_latent(path):
    return torch.Tensor(np.load(path)).to("cuda").float().squeeze().unsqueeze(0)


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

    def create_morphed_latent(self, start_latent, json_info):
        morphed_latent = start_latent.clone()

        for name, sensitivities in cfg.interfacegan_direction_sensitivities.items():
            if name in json_info.keys():
                morph_direction       = load_latent(os.path.join(cfg.latent_directions_path, name + '.npy'))
                morph_sensitivity     = np.abs(sensitivities[0] if (json_info[name] < 0) else sensitivities[1])

                morphed_latent += json_info[name] * morph_sensitivity * morph_direction
        
        return morphed_latent

    def make_interpolation_video(self, latent, img_name, json_info, outdir, aligned_HD_img_path):

        # Set the starting face:
        if cfg.start_face_dir:
            encoded_face_paths = [os.path.join(cfg.start_face_dir, f) for f in os.listdir(cfg.start_face_dir) if ".npy" in f]
            random_face_w      = load_latent(random.choice(encoded_face_paths))
        else:
            random_face_z = torch.from_numpy(np.random.normal(loc=0.0, scale=1.0, size=(1,512))).to("cuda").float()
            random_face_w = self.net.decoder.get_latent(random_face_z).unsqueeze(1).repeat(1, 18, 1)

        trajectory = []
        [trajectory.append(random_face_w) for i in range(int(cfg.freeze_start * cfg.video_out_fps))]

        # Part I: random face ---> target face
        for interpolation_f in np.linspace(0, 1, int(cfg.random_to_target_duration * cfg.video_out_fps)):
            v = (1-interpolation_f) * random_face_w + interpolation_f * latent
            trajectory.append(v)

        [trajectory.append(latent) for i in range(int(cfg.freeze_middle * cfg.video_out_fps))]

        # Part II: target face ---> morphed target face
        morphed_latent = self.create_morphed_latent(latent, json_info)
        for interpolation_f in np.linspace(0, 1, int(cfg.target_to_morphed_duration * cfg.video_out_fps)):
            v = (1-interpolation_f) * latent + interpolation_f * morphed_latent
            trajectory.append(v)

        [trajectory.append(morphed_latent) for i in range(int(cfg.freeze_end * cfg.video_out_fps))]
        final_w_trajectory = torch.stack(trajectory).cpu().numpy()

        # Smooth the final trajectory:
        smoothing_w = int(cfg.smoothing_window * cfg.video_out_fps)
        final_w_trajectory = scipy.ndimage.gaussian_filter(final_w_trajectory, [smoothing_w,0,0,0], mode= 'nearest')

        # Render the frames:
        interpolation_images = self.editor.encode_latent_trajectory(torch.from_numpy(final_w_trajectory).to("cuda").float())
        results = np.stack([np.array(res) for res in interpolation_images])
        
        if cfg.concat_source:
            source_image = np.array(Image.open(aligned_HD_img_path))[np.newaxis, :]
            source_image = np.repeat(source_image, results.shape[0], axis=0)
            results = np.concatenate((source_image, results), axis=cfg.concat_direction)

        video_path = os.path.join(outdir, img_name + '.mp4')
        write_video(video_path, results, cfg.video_out_fps)


    def save_sbs(self, result_image, source_image, output_dir, img_name):
        res = np.concatenate([np.array(source_image),
                              np.array(result_image)], axis=1)
        img = Image.fromarray(res)
        img.save(os.path.join(output_dir, img_name))
        return

def smooth_w(w_in, smoothing_f):
    w_orig_length = w_in.shape[0]
    #w_initial_mean = np.mean(w_in, axis=0)

    print("Smoothing kernel_width: %d   (%.1f%% of total signal length)" %(smoothing_f, 100*smoothing_f/w_orig_length))

    #duplication_samples = int(min(6*smoothing_f, len(w_in))/2)*2
    #w_in = np.append(w_in, w_in[:duplication_samples], axis=0)

    w_in = scipy.ndimage.gaussian_filter(w_in, [smoothing_f,0,0,0], mode= 'nearest')
    #w_in = w_in[int(duplication_samples/2):int(duplication_samples/2)+w_orig_length]

    #Renormalize:
    #w_in = w_in * (w_initial_mean[np.newaxis,:] / np.mean(w_in, axis=0)[np.newaxis,:])

    return w_in

def load_face_latents(folder):
    latent_names = [''.join(f.split('.')[:-1]) for f in os.listdir(folder) if '.npy' in f]
    latent_paths = [os.path.join(folder, f) for f in os.listdir(folder) if '.npy' in f]
    latents = []
    for f in latent_paths:
        latent = np.load(f)
        latent = torch.Tensor(latent).to("cuda")
        latents.append(latent)
    return torch.stack(latents), latent_names