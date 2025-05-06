"""
Helper utilities for the baby cry detector application.
"""
import os
import time
import sys
import gradio as gr
from typing import List, Tuple, Dict, Any, Optional, Union

def format_file_list(files: List[str], current_index: int = 0, playback_mode: str = "random", 
                  max_files: int = 5) -> str:
    """
    Format a list of files into a user-friendly string.
    
    Args:
        files: List of file paths
        current_index: Current index for sequential playback
        playback_mode: Playback mode (random, sequential, all)
        max_files: Maximum number of files to list
        
    Returns:
        str: Formatted file list string
    """
    file_text = f"Audio files ({len(files)}):"
    if not files:
        return file_text + " None"
    
    for i, file in enumerate(files):
        if i < max_files:
            filename = os.path.basename(file)
            next_indicator = " (next)" if i == current_index and playback_mode == "sequential" else ""
            file_text += f"\n - {filename}{next_indicator}"
    
    if len(files) > max_files:
        file_text += f"\n - ... and {len(files) - max_files} more"
    
    return file_text

def process_uploaded_files(files: Union[str, List[str]]) -> List[str]:
    """
    Process uploaded files and return list of file paths.
    
    Args:
        files: Single file path or list of file paths
        
    Returns:
        list: List of file paths
    """
    if not files:
        return []
    
    if isinstance(files, str):  # Single file
        return [files]
    else:  # Multiple files
        return list(files)

def add_files_to_list(files: Union[str, List[str]], existing_files: List[str]) -> Tuple[str, List[str]]:
    """
    Add files to an existing list, avoiding duplicates.
    
    Args:
        files: Files to add
        existing_files: Existing list of files
        
    Returns:
        tuple: (Status message, Updated file list)
    """
    if not files:
        return "No files selected", existing_files
    
    # Convert to list if it's a single file
    file_list = process_uploaded_files(files)
    
    # Add new files, avoiding duplicates
    added_count = 0
    for file in file_list:
        if file and file not in existing_files:
            existing_files.append(file)
            added_count += 1
    
    return f"Added {added_count} file(s)", existing_files

def remove_file_from_list(file_index: Union[str, int], file_list: List[str]) -> Tuple[str, List[str]]:
    """
    Remove a file by index from a list.
    
    Args:
        file_index: Index of file to remove
        file_list: List of files
        
    Returns:
        tuple: (Status message, Updated file list)
    """
    try:
        index = int(file_index)
        if 0 <= index < len(file_list):
            removed = file_list.pop(index)
            return f"Removed: {os.path.basename(removed)}", file_list
        else:
            return "Invalid file index", file_list
    except ValueError:
        return "Invalid selection", file_list

def clear_files(file_list: List[str]) -> Tuple[str, List[str]]:
    """
    Clear all files from a list.
    
    Args:
        file_list: List of files to clear
        
    Returns:
        tuple: (Status message, Empty list)
    """
    count = len(file_list)
    return f"Cleared {count} file(s)", []

def print_system_info() -> None:
    """Print system information for debugging."""
    try:
        import numpy as np
        import pygame
        import librosa
        import sounddevice as sd
        
        print(f"Python version: {sys.version}")
        print(f"Gradio version: {gr.__version__}")
        print(f"NumPy version: {np.__version__}")
        print(f"Pygame version: {pygame.version.ver}")
        print(f"Librosa version: {librosa.__version__}")
        
        # Print sounddevice info
        print(f"Available audio devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f" - Input: {device['name']} (ID: {i})")
            if device['max_output_channels'] > 0:
                print(f" - Output: {device['name']} (ID: {i})")
    except Exception as e:
        print(f"Error printing system info: {e}")