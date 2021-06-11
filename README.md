# FakeMeDeep

A simple codebase to run a LIVE, StyleGAN-based facemorph server.

## Main WorkFlow:
1. edit config.py with the desired parameters
2. run ```python server.py```
3. Any image added to the cfg.input_folder will be encoded into StyleGAN and rendered into a morphing video. If a .json file with the same name as the image is found in cfg.input_folder, it will be used to extract the morph settings. Otherwise default.json is used!

## Things to keep in mind:
- This repo expects all incoming face images to have a unique name (eg add a timestamp)
- There are a ton of configuration options in config.py and the json file
- run python download.py to manually download the pretrained models to your local disk (will be called automatically if you run server.py)
