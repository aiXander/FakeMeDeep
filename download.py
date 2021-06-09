import os, subprocess

def get_download_model_command(file_id, file_name, model_dir):
    """ Get wget download command for downloading the desired model and save to directory pretrained_models. """
    url = r"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={FILE_ID}" -O {SAVE_PATH}/{FILE_NAME} && rm -rf /tmp/cookies.txt""".format(FILE_ID=file_id, FILE_NAME=file_name, SAVE_PATH=model_dir)
    return url

def maybe_download_models():
    model_dir = "encoder4editing/pretrained_models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if "e4e_ffhq_encode.pt" not in os.listdir(model_dir):
        print("*************************************************")
        print("*********** Downloading e4e model... ************")
        print("*************************************************")
        cmd = get_download_model_command("1cUv_reLE6k3604or78EranS7XzuVMWeO", "e4e_ffhq_encode.pt", model_dir)
        os.system(cmd)

    if "shape_predictor_68_face_landmarks.dat" not in os.listdir(model_dir):
        print("*************************************************")
        print("********* Downloading shape predictor... ********")
        print("*************************************************")
        cmd1 = "wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -P %s" %model_dir
        cmd2 = "bzip2 -dk %s/shape_predictor_68_face_landmarks.dat.bz2" %model_dir
        cmd3 = "rm %s/shape_predictor_68_face_landmarks.dat.bz2" %model_dir
        subprocess.call(cmd1.split(" "))
        subprocess.call(cmd2.split(" "))
        subprocess.call(cmd3.split(" "))