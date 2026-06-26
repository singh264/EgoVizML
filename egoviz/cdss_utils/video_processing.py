import logging
import os
import math
from math import floor

import cv2
from moviepy.editor import VideoFileClip
from egoviz.egomodelkit_progress import emit_progress

# Configure the logger
logging.basicConfig(
    level=logging.INFO,  # Set the desired logging level (e.g., INFO, DEBUG)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def segments(duration, n=60):
    """
    Calculate start and end times for each subclip
    """
    return [(i, i + n) for i in range(0, duration, n)]


def save_subclips(
    video_path,
    root,
    n = 60,
    fps = 10,
    frame_fps = 2,
    *,
    frames_done = 0,
    total_frames = 0,
):    
    """
    Save subclips of a video
    """
    logger.info(f"Processing file: {video_path}")
    try:
        # Load video
        clip = VideoFileClip(video_path)

        # Get duration and start and end times
        duration = int(floor(clip.duration))
        times = segments(duration, n)

        # Create subclips and save
        subclip_folder = os.path.join(root, "subclips")
        os.makedirs(subclip_folder, exist_ok=True)  # Ensure the folder exists

        for idx, (start, end) in enumerate(times):
            subclip = clip.subclip(start, end)
            base_name = os.path.splitext(video_path)[0].rstrip(".")
            clip_name = f"{base_name}--{idx + 1}.MP4"
            clip_path = os.path.join(subclip_folder, clip_name)

            logger.info(f"Saving subclip: {clip_path}")

            # Save videos with specified fps and no audio
            subclip.write_videofile(clip_path, fps=fps, audio=False)

            # Create a folder for frames
            frame_output_folder = os.path.splitext(clip_path)[0]
            os.makedirs(frame_output_folder, exist_ok=True)

            # Split subclip into frames and save
            written = frame_split(clip_path, frame_output_folder, fps = frame_fps)
            frames_done += written

            emit_progress(
                "adl_frame_extracted",
                current = frames_done,
                total = total_frames,
            )

        clip.close()
        
        return frames_done
    except Exception as e:
        logger.error(f"Error: {e} for file: {video_path}")

def frame_split(video_path, output_folder, fps=2):
    """ Split video into frames and return the number of frames written. """
    logger.info(f"Splitting frames for: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps_original = int(cap.get(cv2.CAP_PROP_FPS))
    downsample = max(1, fps_original // fps)
    idx = 0
    written = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if idx % downsample == 0:
            outpath = os.path.join(output_folder, f"frame_{idx}.jpg")
            cv2.imwrite(outpath, frame)
            written += 1

        idx += 1

    cap.release()
    cv2.destroyAllWindows()

    return written

def _clean_video_name(file_name):
    stem, suffix = os.path.splitext(file_name)
    
    return f"{stem.rstrip('.')}{suffix}"

def _video_duration_seconds(video_path):
    clip = VideoFileClip(video_path)

    try:
        return int(floor(clip.duration))
    finally:
        clip.close()

def _planned_frame_count(duration, subclip_length, frame_fps):
    subclip_count = len(segments(duration, subclip_length))

    return subclip_count * int(math.ceil(subclip_length * frame_fps))

def process_videos_in_folder(
    dirpath, subclip_length=60, fps=10, frame_fps=2, frames_only=False
):
    """ Split videos into subclips and frames with global progress across all videos. """
    video_files = [
        file
        for file in os.listdir(dirpath)
        if file.upper().endswith(".MP4")
    ]

    video_infos = []

    for file in video_files:
        video_path = os.path.join(dirpath, file)
        duration = _video_duration_seconds(video_path)

        video_infos.append(
            {
                "file": file,
                "duration": duration,
                "planned_frames": _planned_frame_count(
                    duration,
                    subclip_length,
                    frame_fps,
                ),
            }
        )

    video_infos.sort(key = lambda info: (-info["duration"], info["file"]))

    total_frames = sum(info["planned_frames"] for info in video_infos)
    frames_done = 0

    for video_index, info in enumerate(video_infos, start=1):
        file = info["file"]
        display_video = f"video{video_index}"

        emit_progress(
            "adl_video_checked",
            current = video_index,
            total = len(video_infos),
            video = _clean_video_name(file),
            displayVideo = display_video,
        )

        os.chdir(dirpath)

        if frames_only:
            frame_output_folder = os.path.splitext(file)[0].rstrip(".")
            os.makedirs(frame_output_folder, exist_ok = True)

            written = frame_split(file, frame_output_folder, fps=frame_fps)
            frames_done += written

            emit_progress(
                "adl_frame_extracted",
                current = frames_done,
                total = total_frames,
            )
            
            logger.info(f"Frames split for video: {file}")
            
            continue

        frames_done = save_subclips(
            file,
            dirpath,
            subclip_length,
            fps,
            frame_fps,
            frames_done = frames_done,
            total_frames = total_frames,
        )

class VideoObject:
    def __init__(self, root, file):
        self.root = root
        self.file = file
        self.video = os.path.join(root, file)

        # Configure the logger
        logging.basicConfig(
            level=logging.INFO,  # Set the desired logging level (e.g., INFO, DEBUG)
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def frame_split(self, fps, dir="frames"):
        cap = cv2.VideoCapture(self.video)
        fps_original = int(cap.get(cv2.CAP_PROP_FPS))
        downsample = fps_original // fps
        if not os.path.isdir(os.path.join(self.root, dir)):
            os.mkdir(os.path.join(self.root, dir))
            print("Made Directory:", os.path.join(self.root, dir))

        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % downsample == 0:
                outpath = os.path.join(
                    self.root, dir, f'{self.file.split(".")[0]}_frame{idx}.jpg'
                )
                cv2.imwrite(outpath, frame)
            idx += 1

        cap.release()
        cv2.destroyAllWindows()

        # log progress
        self.logger.info(f"Frames split for video: {self.video}")
