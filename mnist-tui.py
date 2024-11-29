import curses
import numpy as np
from PIL import Image
import os
from datetime import datetime

class MNISTCrayonTUI:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.canvas_size = 16
        self.grid_size = 34
        self.canvas = np.zeros((self.canvas_size, self.canvas_size), dtype=int)
        self.grid = []
        self.cursor = [0, 0]
        self.mode = "DRAW"  # DRAW or NAVIGATE
        self.messages = []
        
        # Setup colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Canvas background
        curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)  # Drawing color
        curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Status messages
        
        # Initialize windows
        self.setup_windows()
        
    def setup_windows(self):
        height, width = self.stdscr.getmaxyx()
        self.canvas_win = curses.newwin(self.canvas_size + 2, self.canvas_size * 2 + 2, 1, 1)
        self.info_win = curses.newwin(10, 30, 1, self.canvas_size * 2 + 4)
        self.msg_win = curses.newwin(5, width - 2, height - 6, 1)
        
    def draw_interface(self):
        self.stdscr.clear()
        self.draw_canvas()
        self.draw_info()
        self.draw_messages()
        self.stdscr.refresh()
        
    def draw_canvas(self):
        self.canvas_win.clear()
        self.canvas_win.box()
        
        for y in range(self.canvas_size):
            for x in range(self.canvas_size):
                char = "██" if self.canvas[y][x] else "  "
                if y == self.cursor[0] and x == self.cursor[1]:
                    self.canvas_win.attron(curses.A_REVERSE)
                self.canvas_win.addstr(y + 1, x * 2 + 1, char)
                if y == self.cursor[0] and x == self.cursor[1]:
                    self.canvas_win.attroff(curses.A_REVERSE)
        
        self.canvas_win.refresh()
        
    def draw_info(self):
        self.info_win.clear()
        self.info_win.box()
        info_text = [
            f"Mode: {self.mode}",
            f"Images: {len(self.grid)}",
            "",
            "Controls:",
            "Arrow keys - Move",
            "Space - Toggle pixel",
            "S - Save to grid",
            "C - Clear canvas",
            "Q - Quit"
        ]
        
        for i, text in enumerate(info_text):
            self.info_win.addstr(i + 1, 1, text)
        
        self.info_win.refresh()
        
    def draw_messages(self):
        self.msg_win.clear()
        self.msg_win.box()
        
        for i, msg in enumerate(self.messages[-3:]):
            self.msg_win.addstr(i + 1, 1, msg, curses.color_pair(3))
        
        self.msg_win.refresh()
        
    def add_message(self, msg):
        self.messages.append(msg)
        if len(self.messages) > 10:
            self.messages.pop(0)
            
    def toggle_pixel(self):
        self.canvas[self.cursor[0]][self.cursor[1]] = not self.canvas[self.cursor[0]][self.cursor[1]]
        
    def clear_canvas(self):
        self.canvas = np.zeros((self.canvas_size, self.canvas_size), dtype=int)
        self.add_message("Canvas cleared")
        
    def save_to_grid(self):
        if len(self.grid) >= self.grid_size * self.grid_size:
            self.add_message("Grid is full!")
            return
            
        self.grid.append(self.canvas.copy())
        self.add_message(f"Added to grid ({len(self.grid)}/{self.grid_size * self.grid_size})")
        self.clear_canvas()
        
        if len(self.grid) >= self.grid_size * self.grid_size:
            self.export_grid()
            
    def export_grid(self):
        grid_image = Image.new('L', (self.grid_size * self.canvas_size, 
                                   self.grid_size * self.canvas_size), 255)
        
        for i, canvas in enumerate(self.grid):
            y = (i // self.grid_size) * self.canvas_size
            x = (i % self.grid_size) * self.canvas_size
            
            # Convert binary canvas to PIL image
            img_data = (1 - canvas) * 255
            cell_image = Image.fromarray(img_data.astype('uint8'))
            grid_image.paste(cell_image, (x, y))
            
        os.makedirs('output', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'output/mnist_grid_{timestamp}.png'
        grid_image.save(filename)
        self.add_message(f"Saved grid to {filename}")
        
    def handle_input(self, key):
        if key == ord('q'):
            return False
            
        if key == ord(' '):
            self.toggle_pixel()
        elif key == ord('s'):
            self.save_to_grid()
        elif key == ord('c'):
            self.clear_canvas()
        elif key in [curses.KEY_UP, curses.KEY_DOWN, curses.KEY_LEFT, curses.KEY_RIGHT]:
            self.move_cursor(key)
            
        return True
        
    def move_cursor(self, key):
        if key == curses.KEY_UP and self.cursor[0] > 0:
            self.cursor[0] -= 1
        elif key == curses.KEY_DOWN and self.cursor[0] < self.canvas_size - 1:
            self.cursor[0] += 1
        elif key == curses.KEY_LEFT and self.cursor[1] > 0:
            self.cursor[1] -= 1
        elif key == curses.KEY_RIGHT and self.cursor[1] < self.canvas_size - 1:
            self.cursor[1] += 1
            
    def run(self):
        curses.curs_set(0)  # Hide cursor
        running = True
        
        while running:
            self.draw_interface()
            key = self.stdscr.getch()
            running = self.handle_input(key)

def main(stdscr):
    app = MNISTCrayonTUI(stdscr)
    app.run()

if __name__ == "__main__":
    curses.wrapper(main)