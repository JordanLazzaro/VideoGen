#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def convert_to_mp4(input_file, output_file):
    """Convert a video file to MP4 format."""
    try:
        command = [
            'ffmpeg',
            '-i', str(input_file),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-y',
            str(output_file)
        ]
        
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return process.returncode == 0
    except Exception as e:
        print(f"Error converting {input_file}: {str(e)}")
        return False

def remove_black_bars(input_file, output_file):
    """Detect and remove black bars from video if they exist."""
    try:
        # First detect black bars
        detect_command = [
            'ffmpeg',
            '-i', str(input_file),
            '-vf', 'cropdetect',
            '-f', 'null',
            '-t', '10',  # Only analyze first 10 seconds for speed
            '-'
        ]
        
        process = subprocess.run(
            detect_command,
            stderr=subprocess.PIPE,
            text=True
        )
        
        crop_lines = [line for line in process.stderr.split('\n') if 'crop=' in line]
        if not crop_lines:
            # No cropping needed, just copy the file
            os.replace(input_file, output_file)
            return True

        # Get crop parameters and apply them
        crop_params = crop_lines[-1].split('crop=')[1].split(' ')[0]
        crop_command = [
            'ffmpeg',
            '-i', str(input_file),
            '-vf', f'crop={crop_params}',
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-y',
            str(output_file)
        ]
        
        process = subprocess.run(
            crop_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        return process.returncode == 0
    except Exception as e:
        print(f"Error removing black bars from {input_file}: {str(e)}")
        return False

def safe_remove(file_path):
    """Safely remove a file if it exists."""
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        print(f"Warning: Failed to remove temporary file {file_path}: {str(e)}")

def process_video(input_file, output_file, convert_only):
    """Process a single video file. Returns (input_file, output_file, task_type, success)."""
    temp_file = None
    try:
        print(f"Processing: {input_file.name}")
        
        # Convert to MP4 if needed
        if input_file.suffix.lower() != '.mp4':
            temp_file = input_file.with_suffix('.mp4.temp')
            if not convert_to_mp4(input_file, temp_file):
                return (input_file, output_file, 'convert', False)
            current_file = temp_file
        else:
            current_file = input_file

        if convert_only:
            os.replace(current_file, output_file)
            return (input_file, output_file, 'convert', True)

        # Remove black bars
        success = remove_black_bars(current_file, output_file)
        return (input_file, output_file, 'crop', success)

    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        return (input_file, output_file, 'error', False)
    
    finally:
        # Clean up temp file if it exists
        if temp_file:
            safe_remove(temp_file)

def main():
    # Simple argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='Directory containing video files')
    parser.add_argument('--convert-only', action='store_true', help='Only convert to MP4')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of worker threads (default: 4)')
    args = parser.parse_args()
    
    # Convert directory to Path object and validate
    dir_path = Path(args.directory)
    if not dir_path.is_dir():
        print(f"Error: Directory '{dir_path}' does not exist")
        sys.exit(1)
    
    # Prepare tasks
    video_extensions = {'.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'}
    tasks = []
    
    # Add conversion tasks
    for file_path in dir_path.iterdir():
        if file_path.suffix.lower() in video_extensions:
            output_path = file_path.with_suffix('.mp4')
            tasks.append((file_path, output_path, True))

    # Add black bar removal tasks if not convert_only
    if not args.convert_only:
        for file_path in dir_path.iterdir():
            if file_path.suffix.lower() == '.mp4':
                temp_path = file_path.with_name(f"temp_{file_path.name}")
                tasks.append((file_path, temp_path, False))

    if not tasks:
        print("No video files found to process.")
        return

    # Process tasks using thread pool
    stats = {'converted': 0, 'cropped': 0, 'failed': 0}
    
    print(f"\nProcessing {len(tasks)} files using {args.workers} workers...")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks and get futures
        futures = [
            executor.submit(process_video, input_file, output_file, convert_only)
            for input_file, output_file, convert_only in tasks
        ]
        
        # Process completed tasks as they finish
        for future in as_completed(futures):
            input_file, output_file, task_type, success = future.result()
            
            if success:
                if task_type == 'convert':
                    stats['converted'] += 1
                    print(f"Converted: {input_file.name}")
                elif task_type == 'crop':
                    # Move the temp file to replace the original
                    os.replace(output_file, input_file)
                    stats['cropped'] += 1
                    print(f"Removed black bars: {input_file.name}")
            else:
                stats['failed'] += 1
                print(f"Failed: {input_file.name}")
                # Clean up any temp files
                safe_remove(output_file)
    
    print("\nProcessing Summary:")
    print(f"Files converted to MP4: {stats['converted']}")
    if not args.convert_only:
        print(f"Files cropped: {stats['cropped']}")
    print(f"Failed operations: {stats['failed']}")

if __name__ == "__main__":
    main()