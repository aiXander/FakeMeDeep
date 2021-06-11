###### README ######
'''
- All images dumped into input_folder should have a unique name (eg timestamp)
- 

'''

############################################################

input_folder  = 'Data/Todo/'   # Input folder where the images are placed
debug_mode    = True          # Set this to True to create a more verbose output and show full stack trace for errors

random_to_target_frames      = 120     # How many frames to render for the morph between random and target face
target_to_morphed_frames     = 100     # How many frames to render per target face and morphed version
video_out_fps = 16      # fps of output video
concat_source = 0       # Add the original input image to the video for comparison
concat_direction = 2    # 1 = vertical, 2 = horizontal


interfacegan_directions = {
        'age':        0.75,
        'gender':     0.75,
}

# Sensitivities in both negative and positive directions:
interfacegan_direction_sensitivities = {
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
latent_directions_path    = "stylegan2directions"
model_dir                 = "encoder4editing/pretrained_models/"
e4e_model_path            = model_dir + "e4e_ffhq_encode.pt"
dlib_shape_predictor_path = model_dir + "shape_predictor_68_face_landmarks.dat"

default_json_path = 'default.json' #Fallback json in case no .json is found for a given image
accepted_image_extensions = ['.jpg', '.jpeg', '.png']