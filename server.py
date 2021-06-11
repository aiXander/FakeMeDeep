import cv2, os, time, shutil, json, errno
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import torch

# Custom imports
from download import *
import config as cfg
from encoder import StyleGAN_Encoder

def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred

class StyleGAN_Server():
	def __init__(self):
		self.input_folder              = cfg.input_folder
		self.debug                     = cfg.debug_mode
		self.accepted_image_extensions = cfg.accepted_image_extensions
		self.interfacegan_directions   = cfg.interfacegan_directions
		self.latent_directions_path    = cfg.latent_directions_path

		self.setup_directories()
		self.model = StyleGAN_Encoder(debug = self.debug)

	def setup_directories(self):
		os.makedirs(self.input_folder,     exist_ok = True)
		self.base_datadir     = Path(self.input_folder).parent
		self.error_folder     = os.path.join(self.base_datadir, "Errors")
		self.processed_folder = os.path.join(self.base_datadir, "Processed")
		os.makedirs(self.error_folder,     exist_ok = True)
		os.makedirs(self.processed_folder, exist_ok = True)

	def run(self):
		print("\n\n#########################################")
		print("\nServer started at %s\n" %str(datetime.now()))

		while True:
			time.sleep(0.25) #Check for new inputs 4 times per second

			raw_image_paths = self.get_input_content()
			if len(raw_image_paths) > 0:
				time.sleep(0.25) # Wait for the .json file to be saved to disk!

				img_path   = raw_image_paths[0] #TODO which order do we use?
				img_name   = Path(img_path).stem
				img_folder = os.path.join(self.processed_folder, Path(img_path).stem)
				json_path  = os.path.join(self.input_folder, img_name + '.json')

				json_info  = self.parse_json(json_path, img_name)
				os.makedirs(img_folder, exist_ok = True)

				if not self.debug:
					try:
						self.process_image(img_name, img_path, img_folder, json_info)
						silentremove(img_path)
						silentremove(json_path)

					except Exception as e:
						print("Error processing %s: %s" %(img_path, str(e)))
						try:
							shutil.rmtree(img_folder)
						except:
							pass
						shutil.move(img_path, os.path.join(self.error_folder, os.path.basename(img_path)))
				else:
					self.process_image(img_name, img_path, img_folder, json_info)
					silentremove(img_path)
					silentremove(json_path)

	def get_input_content(self):
		image_paths = sorted([os.path.join(self.input_folder, f) 
			for f in os.listdir(self.input_folder) 
			if f.endswith(tuple(self.accepted_image_extensions))])

		if self.debug:
			print("Found %d images ready for encoding!" %len(image_paths))

		return image_paths

	def parse_json(self, json_path, img_name):
		if not os.path.exists(json_path):
			json_path = cfg.default_json_path
			print("No json file found for %s, using default.json!" %img_name)

		json_info = json.load(open(json_path,))
		if isinstance(json_info, list):
			json_info = json_info[0]

		return json_info

	def process_image(self, img_name, img_path_orig, img_folder, json_info):
		print("Processing %s..." %img_path_orig)
		img_path_new = os.path.join(img_folder, os.path.basename(img_path_orig))
		shutil.copy(img_path_orig, img_path_new)

		# extract face
		aligned_HD_img_path, aligned_LD_img_path = self.model.align_one_img(img_name, img_path_new, fix = 1)

		# encode face
		latent = self.model.predict_latents(aligned_HD_img_path, aligned_LD_img_path, img_name).unsqueeze(0)

		# create videos
		video_dir = os.path.join(img_folder, 'videos')
		os.makedirs(video_dir, exist_ok = True)

		for direction_name, latent_range in self.interfacegan_directions.items():
			if self.debug:
				print("Rendering video for %s..." %direction_name)
			direction_path = os.path.join(self.latent_directions_path, direction_name + '.npy')
			direction      = torch.from_numpy(np.load(direction_path)).cuda().float()

			self.model.make_interpolation_video(latent, img_name, json_info, video_dir, aligned_HD_img_path)

		if self.debug:
			print("Done!\n")

		return

if __name__ == "__main__":

	"""

	cd /home/rednax/Desktop/GitHub_Projects/FakeMeDeep
	python3 server.py

	"""
	maybe_download_models(verbose = cfg.debug_mode)

	print("Starting...")
	SG_server = StyleGAN_Server()
	SG_server.run()