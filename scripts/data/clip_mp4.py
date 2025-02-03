import subprocess
from pathlib import Path
import argparse
from tqdm.auto import tqdm

def clip_video(input_file, output_file, start_time, end_time):
    """
    Clip a video file using ffmpeg with precise cutting.
    Start and end time can be in format: HH:MM:SS.mmm, MM:SS.mmm, or SS.mmm
    """
    try:
        cmd = [
            'ffmpeg',
            '-ss', str(start_time),
            '-i', str(input_file),
            '-to', str(end_time),
            '-c', 'copy',
            str(output_file)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Successfully created clip: {output_file}")
        
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error processing {input_file}: {e.stderr.decode()}")
    except Exception as e:
        print(f"An error occurred processing {input_file}: {str(e)}")

if __name__ == "__main__":
    videos = [
        {
            "in_path":'VideoGameGen/data/full/156_0_Game_Boy_Longplay_-_Kirbys_Dream_Land_-_US.mp4',
            "out_path": "VideoGameGen/data/clipped/156_0_Game_Boy_Longplay_-_Kirbys_Dream_Land_-_US_clipped.mp4",
            "start": "00:40",
            "end": "46:37"
        },
        {
            "in_path": "VideoGameGen/data/full/196_0_Gameboy-Longplay-052-Kirbys-Dreamland-2.mp4",
            "out_path": "VideoGameGen/data/clipped/196_0_Gameboy-Longplay-052-Kirbys-Dreamland-2_clipped.mp4",
            "start": "00:39",
            "end": "01:42:18"
        },
        {
            "in_path": "VideoGameGen/data/full/231_0_Gameboy_Longplay_200_The_Legend_of_Zelda_Links_Awakening.mp4",
            "out_path": "VideoGameGen/data/clipped/231_0_Gameboy_Longplay_200_The_Legend_of_Zelda_Links_Awakening_clipped.mp4",
            "start": "00:06:07",
            "end": "04:42:13"
        },
        {
            "in_path": "VideoGameGen/data/full/232_0_Game_Boy_Longplay_-_Super_Mario_Land_2_-_6_Golden_Coins_-_EN-NA.mp4",
            "out_path": "VideoGameGen/data/clipped/232_0_Game_Boy_Longplay_-_Super_Mario_Land_2_-_6_Golden_Coins_-_EN-NA_clipped.mp4",
            "start": "00:00:15",
            "end": "02:49:26"
        },
        {
            "in_path": "VideoGameGen/data/full/244_0_Gameboy_Longplay_197_Donkey_Kong_Land_A.mp4",
            "out_path": "VideoGameGen/data/clipped/244_0_Gameboy_Longplay_197_Donkey_Kong_Land_A_clipped.mp4",
            "start": "00:00:46",
            "end": "01:09:46"
        },
        {
            "in_path": "VideoGameGen/data/full/247_0_Game_Boy_Longplay_195_Pokemon_Blue.mp4",
            "out_path": "VideoGameGen/data/clipped/247_0_Game_Boy_Longplay_195_Pokemon_Blue_clipped.mp4",
            "start": "00:01:56",
            "end": "06:54:31"
        },
        {
            "in_path": "VideoGameGen/data/full/308_0_Game Boy Longplay [004] Super Mario Land 3 Wario Land.mp4",
            "out_path": "VideoGameGen/data/clipped/308_0_Game Boy Longplay [004] Super Mario Land 3 Wario Land_clipped.mp4",
            "start": "00:01:08",
            "end": "02:35:29"
        },
        {
            "in_path": "VideoGameGen/data/full/313_0_Game Boy Longplay [002] Donkey Kong_.mp4",
            "out_path": "VideoGameGen/data/clipped/313_0_Game Boy Longplay [002] Donkey Kong_clipped.mp4",
            "start": "00:02:03",
            "end": "01:30:12"
        },
        {
            "in_path": "VideoGameGen/data/full/315_0_Game Boy Longplay [001] Super Mario Land.mp4",
            "out_path": "VideoGameGen/data/clipped/315_0_Game Boy Longplay [001] Super Mario Land_clipped.mp4",
            "start": "00:10",
            "end": "25:53"
        },
        {
            "in_path": "VideoGameGen/data/full/325_0_Gameboy_Longplay_157_Tetris.mp4",
            "out_path": "VideoGameGen/data/clipped/325_0_Gameboy_Longplay_157_Tetris_clipped.mp4",
            "start": "00:19",
            "end": "50:19"
        },
        {
            "in_path": "VideoGameGen/data/full/350_0_Game_Boy_Longplay_145_Super_Mario_4.mp4",
            "out_path": "VideoGameGen/data/clipped/350_0_Game_Boy_Longplay_145_Super_Mario_4.mp4",
            "start": "00:18",
            "end": "18:50"
        },
        # {
        #     "in_path": "",
        #     "out_path": "",
        #     "start": "",
        #     "end": ""
        # },
    ]

    for video in tqdm(videos):
        clip_video(
            input_file=video['in_path'],
            output_file=video['out_path'],
            start_time=video['start'],
            end_time=video['end']
        )