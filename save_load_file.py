import pickle
from util import printProgressBar
from moviepy.editor import VideoFileClip

MODEL_PICKLE_FILENAME = 'Model.p'

def save_model(svc, orient, pix_per_cell, cell_per_block):
    print('Saving model to pickle file...')
    try:
        with open(MODEL_PICKLE_FILENAME, 'wb') as pfile:
            pickle.dump(
                {
                    'svc':svc,
                    'orient': orient,
                    'pix_per_cell': pix_per_cell,
                    'cell_per_block': cell_per_block,
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', MODEL_PICKLE_FILENAME, ':', e)
        raise

    print('Model cached in pickle file:', MODEL_PICKLE_FILENAME)

def load_model():
    print('Loading model from pickle file:', MODEL_PICKLE_FILENAME, '...')
    with open(MODEL_PICKLE_FILENAME, mode='rb') as file:
        data = pickle.load(file)

    svc = data['svc']
    orient = data['orient']
    pix_per_cell = data['pix_per_cell']
    cell_per_block = data ['cell_per_block']

    print('Loaded model')
    return svc, orient, pix_per_cell, cell_per_block

def load_frames(video_path = "project_video.mp4", start_frame = None, end_frame = None):
    # The file referenced in clip1 is the original video before anything has been done to it
    input = VideoFileClip(video_path)

    len_frames = int(input.fps * input.duration)
    len_frames = len_frames if end_frame == None or end_frame > len_frames else end_frame
    i = 0
    # Initial call to print 0% progress
    printProgressBar(0, len_frames, 'Loading frames: ' + video_path)
    frames = []
    for frame in input.iter_frames():
        if start_frame == None or i > start_frame:
            frames.append(frame)
            # Update Progress Bar
            printProgressBar(i+1, len_frames, 'Loading frames: ' + video_path)
            if i - 1 >= len_frames:
                break
        i = i+1

    return frames, input.fps

def save_frames(frames, fps, output_path = "output.mp4"):
    from moviepy.editor import ImageSequenceClip
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_path, audio=False)
