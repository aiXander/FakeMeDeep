
############################################################

input_folder    = 'data/todo/'   # Input folder where the images are placed
debug_mode      = False           # Set to True for verbose output + full error stack traces (server will crash on any errors!!)

#start_face_dir  = 'famous_faces' # Path of a directory with encoded faces (.npy files), set to None to start from random faces
start_face_dir  = None            # Path of a directory with encoded faces (.npy files), set to None to start from random faces

if debug_mode:
        random_to_target_duration      = 5.0     # How many frames to render for the morph between random and target face
        target_to_morphed_duration     = 5.0     # How many frames to render per target face and morphed version
        video_out_fps = 16      # fps of output video
        concat_source = 1       # Add the original input image to the video for comparison
        concat_direction = 2    # 1 = vertical, 2 = horizontal
else:
        random_to_target_duration      = 10.0    # Duration of the morph between random and target face (seconds)
        target_to_morphed_duration     = 7.50    # Duration of the morph between target face and morphed version (seconds)
        video_out_fps = 24      # fps of output video
        concat_source = 1       # Add the original input image to the video for comparison
        concat_direction = 2    # 1 = vertical, 2 = horizontal

freeze_start  = 1.0 # How long to freeze at the start  of each video (seconds)
freeze_middle = 0.7 # How long to freeze at the middle of each video (seconds)
freeze_end    = 2.0 # How long to freeze at the end    of each video (seconds)

 # Width of the smoothing window applied to the final interpolation (seconds)
smoothing_window = 0.33 # 0 ---> abrupt changes between phases of the interpolation

#Sometimes the face detection fails, in this case you can eg use the raw input image (at your own risk)
use_raw_img_if_face_detection_fails = False

# Sensitivities in both negative and positive directions:
interfacegan_direction_sensitivities = {
        'age':                  [-7, 8],
        #'eye_distance':         [-24, 24],
        'eyebrow':              [-24, 24],
        #'eye_ratio':            [-18, 18],
        'eyes_open':            [-30, 15],
        'gender':               [-7, 7],
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


# Fixed, pretrained model paths:
latent_directions_path    = "stylegan2directions"
model_dir                 = "encoder4editing/pretrained_models/"
e4e_model_path            = model_dir + "e4e_ffhq_encode.pt"
dlib_shape_predictor_path = model_dir + "shape_predictor_68_face_landmarks.dat"

default_json_path = 'default.json' #Fallback json in case no .json is found for a given image
accepted_image_extensions = ['.jpg', '.jpeg', '.png']