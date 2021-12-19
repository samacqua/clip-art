import numpy as np
from tqdm import tqdm
from PIL import Image
from subprocess import Popen, PIPE
import os

def gen_video(frames, min_fps=20, max_fps=60, duration=10, fname='video.mp4'):
    fps = np.clip(len(frames)/duration,min_fps,max_fps)

    p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(fps), '-i', '-', '-vcodec', 'libx264', '-r', str(fps), '-pix_fmt', 'yuv420p', '-crf', '17', '-preset', 'veryslow', fname], stdin=PIPE)

    for im in tqdm(frames):
        im.save(p.stdin, 'PNG')
    p.stdin.close()
    p.wait()


def gen_video_from_fnames(fnames, min_fps=20, max_fps=60, duration=10, fname='video.mp4'):
    frames = [Image.open(f) for f in fnames]
    gen_video(frames, min_fps, max_fps, duration, fname)


def compose(funcs, args, name, base_dir='clip-art'):
    """
    compose different CLIP art functions into one continuous creation
    :param functions: list of functions taking only one argument (init_image) that runs the desired CLIP-art function
        when called
    """
    assert len(funcs) == len(args)
    name = name.replace(' ', '_')
    composition_dir = os.path.join(base_dir, name)
    os.makedirs(composition_dir, exist_ok=True)

    last_image = None
    frames = []
    for i, (func, model_args) in enumerate(zip(funcs, args)):
        out_fnames = func(init_image=last_image, **model_args)
        last_image = out_fnames[-1]
        iter_frames = [Image.open(f) for f in out_fnames]
        frames += iter_frames

        gen_video(iter_frames, fname=os.path.join(composition_dir, f'{name}_{i}.mp4'))

    gen_video(frames, fname=os.path.join(composition_dir, f'{name}.mp4'))
    return frames


def sequential(funcs, args, names=None, base_dir='clip-art'):
    """
    run many different clip experiments
    :param functions: list of functions taking only one argument (init_image) that runs the desired CLIP-art function
        when called
    """
    names = names if names is not None else [f'video_{i}' for i in range(len(funcs))]
    assert len(funcs) == len(args) == len(names)

    for func, model_args, name in zip(funcs, args, names):
        out_fnames = func(**model_args)
        iter_frames = [Image.open(f) for f in out_fnames]

        composition_dir = os.path.join(base_dir, name)
        os.makedirs(composition_dir, exist_ok=True)

        gen_video(iter_frames, fname=os.path.join(composition_dir, f'{name}.mp4'))
