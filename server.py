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
		self.latent_directions_path    = cfg.latent_directions_path

		self.setup_directories()
		self.model = StyleGAN_Encoder(debug = self.debug)

	def setup_directories(self):
		os.makedirs(self.input_folder,     exist_ok = True)
		self.base_datadir     = Path(self.input_folder).parent
		self.error_folder     = os.path.join(self.base_datadir, "errors")
		self.processed_folder = os.path.join(self.base_datadir, "processed")
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
				print("\n --- Processing %s..." %img_name)

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

	def process_image_without_face_detection(self, img_path_new, img_name, img_folder):
		print("Face extraction for %s failed, using raw image instead!" %img_name)
		img = cv2.imread(img_path_new)
		img = cv2.resize(img, (1024, 1024))

		aligned_HD_img_path = os.path.join(img_folder, '%s_aligned_HD.jpg' %img_name)
		cv2.imwrite(aligned_HD_img_path, img)

		aligned_LD_img_path = os.path.join(img_folder, '%s_aligned_LD.jpg' %img_name)
		img = cv2.resize(img, (256, 256))
		cv2.imwrite(aligned_LD_img_path, img)

		return aligned_HD_img_path, aligned_LD_img_path


	def process_image(self, img_name, img_path_orig, img_folder, json_info):
		img_path_new = os.path.join(img_folder, os.path.basename(img_path_orig))
		shutil.copy(img_path_orig, img_path_new)

		# extract face
		if cfg.use_raw_img_if_face_detection_fails:
			try:
				aligned_HD_img_path, aligned_LD_img_path = self.model.align_one_img(img_name, img_path_new, fix = 1)
			except:
				aligned_HD_img_path, aligned_LD_img_path = self.process_image_without_face_detection(img_path_new, img_name, img_folder)

		else:
			aligned_HD_img_path, aligned_LD_img_path = self.model.align_one_img(img_name, img_path_new, fix = 1)

		# encode face
		latent = self.model.predict_latents(aligned_HD_img_path, aligned_LD_img_path, img_name).unsqueeze(0)

		# create video
		self.model.make_interpolation_video(latent, img_name, json_info, img_folder, aligned_HD_img_path)

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
	with torch.no_grad():
		SG_server = StyleGAN_Server()
		SG_server.run()