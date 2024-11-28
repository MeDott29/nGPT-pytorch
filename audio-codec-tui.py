import numpy as np
import scipy.io.wavfile as wav
from scipy.fft import fft, ifft
import curses
import logging
from dataclasses import dataclass
import os
from typing import Dict, Any, List, Optional
from PIL import Image
from pathlib import Path

@dataclass
class CodecConfig:
    sample_rate: int = 44100
    min_freq: int = 20
    max_freq: int = 20000
    compression_factor: int = 4
    partial_threshold: float = 0.1

class FileSelector:
    def __init__(self, stdscr, start_path: str = "."):
        self.stdscr = stdscr
        self.current_path = Path(start_path).resolve()
        self.files = []
        self.current_index = 0
        self.offset = 0
        self.refresh_files()
        
    def refresh_files(self):
        try:
            self.files = [".."] + sorted([f.name for f in self.current_path.iterdir()])
            self.current_index = 0
            self.offset = 0
        except Exception as e:
            self.files = [".."]
            
    def display(self):
        height, width = self.stdscr.getmaxyx()
        max_display = height - 8
        
        # Show current path
        self.stdscr.addstr(2, 2, f"Location: {self.current_path}")
        
        # Show files
        for i in range(max_display):
            idx = i + self.offset
            if idx >= len(self.files):
                break
                
            file_name = self.files[idx]
            full_path = self.current_path / file_name
            
            # Determine if it's a directory
            is_dir = file_name == ".." or os.path.isdir(full_path)
            display_name = f"/{file_name}" if is_dir else file_name
            
            if idx == self.current_index:
                self.stdscr.attron(curses.color_pair(1))
                self.stdscr.addstr(4 + i, 2, f"> {display_name}")
                self.stdscr.attroff(curses.color_pair(1))
            else:
                self.stdscr.addstr(4 + i, 4, display_name)
    
    def handle_input(self, key) -> Optional[str]:
        logging.debug(f"Key pressed: {key}, Current index: {self.current_index}")
        if key == curses.KEY_UP:
            self.current_index = max(0, self.current_index - 1)
            if self.current_index < self.offset:
                self.offset = self.current_index
        elif key == curses.KEY_DOWN:
            self.current_index = min(len(self.files) - 1, self.current_index + 1)
            height = self.stdscr.getmaxyx()[0]
            if self.current_index >= self.offset + height - 8:
                self.offset = self.current_index - height + 9
        elif key == ord('\n'):
            selected = self.files[self.current_index]
            full_path = self.current_path / selected
            
            if selected == "..":
                self.current_path = self.current_path.parent
                self.refresh_files()
            elif os.path.isdir(full_path):
                self.current_path = full_path
                self.refresh_files()
            else:
                return str(full_path)
        return None

class Menu:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.current_option = 0
        self.options = [
            "Compress Audio File",
            "Convert Image to Audio",
            "Settings",
            "Quit"
        ]
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        self.stdscr.keypad(1)
        
    def display(self):
        self.stdscr.clear()
        height, width = self.stdscr.getmaxyx()
        
        title = "Audio Codec"
        x = (width - len(title)) // 2
        self.stdscr.attron(curses.A_BOLD)
        self.stdscr.addstr(2, x, title)
        self.stdscr.attroff(curses.A_BOLD)
        
        for idx, option in enumerate(self.options):
            y = height//2 - len(self.options)//2 + idx
            x = (width - len(option)) // 2
            
            if idx == self.current_option:
                self.stdscr.attron(curses.color_pair(1))
                self.stdscr.addstr(y, x-2, "> " + option)
                self.stdscr.attroff(curses.color_pair(1))
            else:
                self.stdscr.addstr(y, x, option)
        
        instructions = "↑↓: Navigate | Enter: Select | q: Quit"
        self.stdscr.addstr(height-2, (width - len(instructions)) // 2, instructions)
        
        self.stdscr.refresh()

    def show_message(self, msg: str, error: bool = False):
        height, width = self.stdscr.getmaxyx()
        color = curses.color_pair(3) if error else curses.color_pair(2)
        self.stdscr.addstr(height-4, (width - len(msg)) // 2, msg, color)
        self.stdscr.refresh()

    def select_file(self, prompt: str, file_type: str = None) -> Optional[str]:
        """
        Select a file with optional type validation
        file_type: 'audio', 'image', or None
        """
        self.show_message(prompt)
        selector = FileSelector(self.stdscr)
        
        while True:
            self.stdscr.clear()
            selector.display()
            self.stdscr.refresh()
            
            key = self.stdscr.getch()
            if key == ord('q'):
                return None
                
            result = selector.handle_input(key)
            if result is not None:
                return result

class AudioProcessor:
    def __init__(self, config: CodecConfig):
        self.config = config
    
    def compress_audio(self, input_path: str, output_path: str, menu: Menu) -> bool:
        try:
            menu.show_message("Loading audio file...")
            sample_rate, audio = wav.read(input_path)
            audio = audio.astype(float) / 32767.0

            menu.show_message("Compressing...")
            frequencies = fft(audio)
            mask = np.abs(np.gradient(frequencies)) > self.config.partial_threshold
            compressed = frequencies * mask
            reconstructed = np.real(ifft(compressed))
            
            menu.show_message("Saving...")
            wav.write(output_path, sample_rate, (reconstructed * 32767).astype(np.int16))
            return True
            
        except Exception as e:
            menu.show_message(f"Error: {str(e)}", error=True)
            return False

    def image_to_audio(self, image_path: str, audio_path: str, menu: Menu) -> bool:
        try:
            menu.show_message("Converting image...")
            img = Image.open(image_path).convert('L')
            img_array = np.array(img) / 255.0
            
            frequencies = fft(img_array.flatten())
            audio_signal = np.real(ifft(frequencies))
            audio_signal = audio_signal / np.max(np.abs(audio_signal))
            
            menu.show_message("Saving audio file...")
            wav.write(audio_path, self.config.sample_rate, 
                     (audio_signal * 32767).astype(np.int16))
            return True
            
        except Exception as e:
            menu.show_message(f"Error: {str(e)}", error=True)
            return False

def main(stdscr):
    curses.curs_set(0)
    menu = Menu(stdscr)
    config = CodecConfig()
    processor = AudioProcessor(config)

    while True:
        menu.display()
        key = stdscr.getch()

        if key == curses.KEY_UP and menu.current_option > 0:
            menu.current_option -= 1
        elif key == curses.KEY_DOWN and menu.current_option < len(menu.options) - 1:
            menu.current_option += 1
        elif key == ord('q'):
            break
        elif key == ord('\n'):
            if menu.current_option == 0:  # Compress Audio
                input_path = menu.select_file("Select input audio file")
                if input_path:
                    output_path = menu.select_file("Select output location")
                    if output_path:
                        if processor.compress_audio(input_path, output_path, menu):
                            menu.show_message("Audio compressed successfully!")
                
            elif menu.current_option == 1:  # Image to Audio
                image_path = menu.select_file("Select input image")
                if image_path:
                    audio_path = menu.select_file("Select output location")
                    if audio_path:
                        if processor.image_to_audio(image_path, audio_path, menu):
                            menu.show_message("Conversion complete!")
                    
            elif menu.current_option == 3:  # Quit
                break

if __name__ == "__main__":
    curses.wrapper(main)