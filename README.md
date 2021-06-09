# FakeMeDeep

A simple codebase to run a LIVE, StyleGAN-based facemorph.

## Main WorkFlow:
1. edit config.py with the desired parameters
2. run ```python SG_server.py```
3. Any image added to the cfg.input_folder will be encoded into StyleGAN and rendered into a set of morphing videos

## Things to keep in mind:
- This repo expects all incoming face images to have a unique name (eg add a timestamp)
