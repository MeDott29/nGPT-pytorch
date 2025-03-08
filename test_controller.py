"""
Xbox Controller Test Script

This script tests if your Xbox controller is properly connected and recognized by pygame.
Run this before using the nGPT Explorer to ensure your controller is working correctly.
"""

import pygame
import sys
import time
import os

def main():
    # Initialize pygame
    try:
        pygame.init()
        pygame.joystick.init()
    except pygame.error as e:
        print(f"Error initializing pygame: {e}")
        sys.exit(1)
    
    # Set up a simple display
    try:
        screen = pygame.display.set_mode((640, 480))
        pygame.display.set_caption("Xbox Controller Test")
        font = pygame.font.SysFont("Arial", 18)
        clock = pygame.time.Clock()
    except pygame.error as e:
        print(f"Error setting up display: {e}")
        pygame.quit()
        sys.exit(1)
    
    # Check for controllers
    joystick_count = pygame.joystick.get_count()
    
    if joystick_count == 0:
        print("\nNo controllers found. Please connect your Xbox controller and try again.")
        print("You can still use the nGPT Explorer with keyboard controls.")
        print("\nPress any key to continue...")
        
        # Display message on screen
        screen.fill((0, 0, 0))
        text1 = font.render("No controllers found!", True, (255, 0, 0))
        text2 = font.render("Please connect your Xbox controller and restart this test.", True, (255, 255, 255))
        text3 = font.render("You can still use the nGPT Explorer with keyboard controls.", True, (255, 255, 255))
        text4 = font.render("Press any key to continue...", True, (255, 255, 0))
        
        screen.blit(text1, (20, 100))
        screen.blit(text2, (20, 140))
        screen.blit(text3, (20, 180))
        screen.blit(text4, (20, 240))
        
        pygame.display.flip()
        
        # Wait for key press
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    waiting = False
            time.sleep(0.1)
        
        pygame.quit()
        return
    
    # Initialize the first controller
    try:
        controller = pygame.joystick.Joystick(0)
        controller.init()
        
        print(f"\nController detected: {controller.get_name()}")
        print(f"Number of axes: {controller.get_numaxes()}")
        print(f"Number of buttons: {controller.get_numbuttons()}")
        print("\nTest your controller by moving sticks and pressing buttons.")
        print("The display will show the current state of your controller.")
        print("Press ESC to exit the test.")
    except pygame.error as e:
        print(f"Error initializing controller: {e}")
        pygame.quit()
        sys.exit(1)
    
    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Clear screen
        screen.fill((0, 0, 0))
        
        # Get controller input
        try:
            axes = [controller.get_axis(i) for i in range(controller.get_numaxes())]
            buttons = [controller.get_button(i) for i in range(controller.get_numbuttons())]
        except pygame.error:
            # Controller might have been disconnected
            screen.fill((0, 0, 0))
            text = font.render("Controller disconnected! Please reconnect and restart the test.", True, (255, 0, 0))
            screen.blit(text, (20, 240))
            pygame.display.flip()
            time.sleep(2)
            running = False
            continue
        
        # Display controller info
        title = font.render(f"Controller: {controller.get_name()}", True, (255, 255, 255))
        screen.blit(title, (20, 20))
        
        # Display axes
        y = 60
        screen.blit(font.render("Axes:", True, (255, 255, 0)), (20, y))
        y += 30
        
        axis_names = [
            "Left Stick X",
            "Left Stick Y",
            "Right Stick X",
            "Right Stick Y",
            "Left Trigger",
            "Right Trigger"
        ]
        
        for i, axis in enumerate(axes):
            name = axis_names[i] if i < len(axis_names) else f"Axis {i}"
            text = font.render(f"{name}: {axis:.3f}", True, (255, 255, 255))
            screen.blit(text, (20, y))
            
            # Draw a bar to visualize the axis value
            bar_x = 300
            bar_width = 200
            bar_height = 15
            
            pygame.draw.rect(screen, (100, 100, 100), (bar_x, y, bar_width, bar_height))
            
            # Map axis value (-1 to 1) to bar position
            value_normalized = (axis + 1) / 2  # Map from [-1, 1] to [0, 1]
            value_width = int(value_normalized * bar_width)
            
            pygame.draw.rect(screen, (0, 255, 0), (bar_x, y, value_width, bar_height))
            
            y += 25
        
        # Display buttons
        y += 20
        screen.blit(font.render("Buttons:", True, (255, 255, 0)), (20, y))
        y += 30
        
        button_names = [
            "A", "B", "X", "Y", 
            "Left Bumper", "Right Bumper", 
            "Back", "Start", 
            "Left Stick Press", "Right Stick Press"
        ]
        
        for i, button in enumerate(buttons):
            name = button_names[i] if i < len(button_names) else f"Button {i}"
            color = (0, 255, 0) if button else (255, 0, 0)
            text = font.render(f"{name}: {'Pressed' if button else 'Released'}", True, color)
            screen.blit(text, (20, y))
            y += 25
        
        # Instructions
        y = 420
        instructions = [
            "Press all buttons and move all sticks to test your controller",
            "Press ESC to exit"
        ]
        
        for instruction in instructions:
            text = font.render(instruction, True, (200, 200, 200))
            screen.blit(text, (20, y))
            y += 25
        
        pygame.display.flip()
        clock.tick(60)
    
    # Clean up
    pygame.quit()
    
    print("\nController test completed.")
    print("If your controller worked correctly, you're ready to use the nGPT Explorer!")
    print("If you had issues, check your controller connection or use keyboard controls.")

if __name__ == "__main__":
    # Clear screen for better visibility
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=== Xbox Controller Test ===")
    main() 