"""
Audio player for playing response sounds when baby crying is detected.
"""
import os
import time
import pygame
import threading
import random
from typing import List, Optional, Callable, Union

class AudioPlayer:
    """Audio player for playing audio files."""
    
    def __init__(self):
        """Initialize the audio player."""
        self.is_playing = False
        self.current_file = None
        self.playback_stop_event = None
        self._init_player()
    
    def _init_player(self) -> bool:
        """
        Initialize pygame mixer.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            pygame.mixer.quit()
            pygame.mixer.init(frequency=44100)
            return True
        except Exception as e:
            print(f"Audio player initialization error: {e}")
            return False
    
    def _select_files_to_play(self, files: List[str], mode: str, current_index: int = 0) -> List[str]:
        """
        Select which audio files to play based on the playback mode.
        
        Args:
            files: List of audio file paths
            mode: Playback mode ("random", "sequential", "all")
            current_index: Current index for sequential mode
            
        Returns:
            list: List of files to play
        """
        if not files:
            return []
            
        if mode == "all":
            # Play all files
            return files
        elif mode == "random":
            # Select a random file
            return [random.choice(files)]
        elif mode == "sequential":
            # Select the next file in sequence
            selected_file = files[current_index % len(files)]
            return [selected_file]
        else:
            # Default to the first file
            return [files[0]]
    
    def play_files(self, files: List[str], mode: str = "random", current_index: int = 0,
                  on_start: Optional[Callable[[str], None]] = None, 
                  on_complete: Optional[Callable[[str], None]] = None,
                  stop_event: Optional[threading.Event] = None) -> None:
        """
        Play audio files based on specified mode.
        
        Args:
            files: List of audio file paths
            mode: Playback mode ("random", "sequential", "all")
            current_index: Current index for sequential mode
            on_start: Callback when playback starts, receives filename
            on_complete: Callback when playback completes, receives status message
            stop_event: Threading event to signal stopping
        """
        if not files:
            if on_complete:
                on_complete("No files to play")
            return
        
        # Create stop event if none provided
        self.playback_stop_event = stop_event or threading.Event()
        
        # Select files based on mode
        files_to_play = self._select_files_to_play(files, mode, current_index)
        
        self.is_playing = True
        
        try:
            if not self._init_player():
                if on_complete:
                    on_complete("Failed to initialize audio player")
                return
            
            for file in files_to_play:
                if self.playback_stop_event.is_set():
                    break
                    
                if not os.path.exists(file):
                    print(f"File not found: {file}")
                    continue
                
                self.current_file = file
                file_basename = os.path.basename(file)
                
                # Call start callback with filename
                if on_start:
                    on_start(file_basename)
                
                pygame.mixer.music.load(file)
                pygame.mixer.music.play()
                
                # Wait for the audio to finish
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                    # Break if stop event is set
                    if self.playback_stop_event.is_set():
                        break
                
                # Small pause between files if playing multiple
                if len(files_to_play) > 1 and not self.playback_stop_event.is_set():
                    time.sleep(0.5)
            
            self.is_playing = False
            self.current_file = None
            
            if on_complete:
                on_complete("Playback completed")
                
        except Exception as e:
            self.is_playing = False
            self.current_file = None
            if on_complete:
                on_complete(f"Audio playback error: {e}")
    
    def stop(self) -> None:
        """Stop any ongoing playback."""
        if self.playback_stop_event:
            self.playback_stop_event.set()
            
        if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            
        self.is_playing = False
        self.current_file = None