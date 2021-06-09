###### README ######
'''
- All images dumped into input_folder should have a unique name (eg timestamp)
- 

'''

'''
import shutil, os
source_img_dir = '/home/rednax/Desktop/GitHub_Projects/FakeMeDeep/Data/raw'
shutil.rmtree('Data2')
os.makedirs('Data2/Todo/', exist_ok = True)
for f in os.listdir(source_img_dir):
    shutil.copy(os.path.join(source_img_dir, f), 'Data2/Todo')
'''


############################################################

input_folder  = 'Data/Todo/'   # Input folder where the images are placed
debug_mode    = False            # Set this to True to create a more verbose output and show full stack trace for errors

n_frames      = 120     # How many frames to render per video
video_out_fps = 16      # fps of output video
concat_source = 0       # Add the original input image to the video for comparison
concat_direction = 2    # 1 = vertical, 2 = horizontal

interfacegan_directions = {
        'age':                  [-3, 6],
        #'eye_distance':         [-24, 24],
        'eyebrow':              [-24, 24],
        #'eye_ratio':            [-18, 18],
        'eyes_open':            [-30, 15],
        'gender':               [-6, 6],
        #'lip_ratio':            [-12, 12],
        'mouth_open':           [-24, 30],
        #'mouth_ratio':          [-20, 20],
        #'nose_mouth_distance':  [-12, 12],
        #'nose_ratio':           [-20, 20],
        #'nose_tip':             [-20, 20],
        'pitch':                [-4, 4],
        #'roll':                 [-20, 20],
        'smile':                [-1.5, 4],
        'yaw':                  [-4.5, 4.5],
}

############################################################


# Fxied, pretrained model paths:
e4e_model_path            = "encoder4editing/pretrained_models/e4e_ffhq_encode.pt"
dlib_shape_predictor_path = "encoder4editing/pretrained_models/shape_predictor_68_face_landmarks.dat"
latent_directions_path    = "stylegan2directions"

accepted_image_extensions = ['.jpg', '.jpeg', '.png']