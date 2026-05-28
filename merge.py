import moviepy.editor as mpe


def merge(mp4_path, wav_path, result_path, fps):
    video = mpe.VideoFileClip(mp4_path)
    video = video.set_audio(mpe.AudioFileClip(wav_path))
    video.write_videofile(result_path, fps=fps)
