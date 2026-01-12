import pygame
import time
from datetime import datetime

class UIManager:
    """
    Professional UI Manager for 3.5" touchscreen display
    Handles all visual feedback and touch interactions
    IMPORTANT: All pygame operations run in the main thread
    """
    
    def __init__(self, primary_color="#c2255c"):
        """
        Initialize the UI with custom branding
        
        Args:
            primary_color: Hex color code for branding (default: #c2255c)
        """
        # Set SDL to use framebuffer for Raspberry Pi
        import os
        os.environ['SDL_VIDEODRIVER'] = 'fbcon'
        os.environ['SDL_FBDEV'] = '/dev/fb1'  # fb1 is typically the LCD
        os.environ['SDL_NOMOUSE'] = '1'  # Disable mouse cursor initially
        
        pygame.init()
        pygame.display.init()
        
        # Display configuration for 3.5" RPi Display (480x320)
        self.width = 480
        self.height = 320
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.mouse.set_visible(True)
        pygame.display.set_caption("Smart Doorbell")
        
        # Color scheme
        self.PRIMARY = self._hex_to_rgb(primary_color)
        self.PRIMARY_DARK = self._darken_color(self.PRIMARY, 0.3)
        self.PRIMARY_LIGHT = self._lighten_color(self.PRIMARY, 0.3)
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (128, 128, 128)
        self.LIGHT_GRAY = (200, 200, 200)
        self.DARK_GRAY = (50, 50, 50)
        self.GREEN = (76, 175, 80)
        self.RED = (244, 67, 54)
        self.YELLOW = (255, 193, 7)
        self.BLUE = (33, 150, 243)
        
        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.font_tiny = pygame.font.Font(None, 18)
        
        # State
        self.current_state = "loading"
        self.state_data = {}
        self._call_button_rect = None
        self.clock = pygame.time.Clock()
        
        # Animation
        self.animation_time = 0
    
    def _hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _darken_color(self, color, factor):
        """Darken a color by a factor (0-1)"""
        return tuple(int(c * (1 - factor)) for c in color)
    
    def _lighten_color(self, color, factor):
        """Lighten a color by a factor (0-1)"""
        return tuple(min(255, int(c + (255 - c) * factor)) for c in color)
    
    def update(self):
        """
        Update and render the UI - MUST be called from main thread
        Returns: touch position if call button was pressed, None otherwise
        """
        self.animation_time += 1
        call_pressed = False
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self._call_button_rect and self._call_button_rect.collidepoint(event.pos):
                    call_pressed = True
                    print("[UI] Call button pressed")
        
        # Render current state
        if self.current_state == "loading":
            self._render_loading()
        elif self.current_state == "idle":
            self._render_idle()
        elif self.current_state == "detecting":
            self._render_detecting()
        elif self.current_state == "access_granted":
            self._render_access_granted()
        elif self.current_state == "access_denied":
            self._render_access_denied()
        elif self.current_state == "calling":
            self._render_calling()
        elif self.current_state == "status":
            self._render_status()
        
        pygame.display.flip()
        self.clock.tick(30)  # 30 FPS
        
        return call_pressed
    
    # ----------------------------
    # State Setters (Thread-safe)
    # ----------------------------
    def show_loading(self, title, message):
        """Display loading screen with progress"""
        self.current_state = "loading"
        self.state_data = {"title": title, "message": message}
    
    def update_loading(self, message, progress=None):
        """Update loading screen message"""
        self.state_data["message"] = message
        if progress is not None:
            self.state_data["progress"] = progress
    
    def show_idle(self, door_locked=True):
        """Display idle/ready state"""
        self.current_state = "idle"
        self.state_data = {"door_locked": door_locked}
    
    def show_detecting(self):
        """Display face detection in progress"""
        self.current_state = "detecting"
        self.state_data = {}
    
    def show_access_granted(self, person_name):
        """Display access granted screen"""
        self.current_state = "access_granted"
        self.state_data = {"person_name": person_name}
    
    def show_access_denied(self):
        """Display access denied screen"""
        self.current_state = "access_denied"
        self.state_data = {}
    
    def show_calling(self):
        """Display calling owner screen"""
        self.current_state = "calling"
        self.state_data = {}
    
    def show_status(self, status_type, message):
        """Display a general status message"""
        self.current_state = "status"
        self.state_data = {"status_type": status_type, "message": message}
    
    # ----------------------------
    # Drawing Utilities
    # ----------------------------
    def _draw_gradient_background(self, color1, color2):
        """Draw a vertical gradient background"""
        for y in range(self.height):
            ratio = y / self.height
            color = tuple(
                int(color1[i] * (1 - ratio) + color2[i] * ratio)
                for i in range(3)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.width, y))
    
    def _draw_header(self, title, subtitle=""):
        """Draw header section with title and subtitle"""
        pygame.draw.rect(self.screen, self.PRIMARY, (0, 0, self.width, 70))
        
        title_text = self.font_medium.render(title, True, self.WHITE)
        title_rect = title_text.get_rect(center=(self.width // 2, 25))
        self.screen.blit(title_text, title_rect)
        
        if subtitle:
            subtitle_text = self.font_tiny.render(subtitle, True, self.PRIMARY_LIGHT)
            subtitle_rect = subtitle_text.get_rect(center=(self.width // 2, 50))
            self.screen.blit(subtitle_text, subtitle_rect)
    
    def _draw_status_indicator(self, x, y, color, label):
        """Draw a small status indicator circle with label"""
        pygame.draw.circle(self.screen, color, (x, y), 8)
        pygame.draw.circle(self.screen, self.WHITE, (x, y), 8, 2)
        
        label_text = self.font_tiny.render(label, True, self.WHITE)
        label_rect = label_text.get_rect(midleft=(x + 15, y))
        self.screen.blit(label_text, label_rect)
    
    def _draw_button(self, x, y, width, height, text, color, text_color=None):
        """Draw a rounded button and return its rect"""
        if text_color is None:
            text_color = self.WHITE
        
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, color, rect, border_radius=10)
        pygame.draw.rect(self.screen, self._darken_color(color, 0.2), rect, 3, border_radius=10)
        
        button_text = self.font_small.render(text, True, text_color)
        text_rect = button_text.get_rect(center=rect.center)
        self.screen.blit(button_text, text_rect)
        
        return rect
    
    def _draw_icon_lock(self, x, y, size, locked=True):
        """Draw a lock icon"""
        color = self.RED if locked else self.GREEN
        
        body_rect = pygame.Rect(x - size//3, y, size//1.5, size//1.5)
        pygame.draw.rect(self.screen, color, body_rect, border_radius=5)
        
        if locked:
            pygame.draw.arc(self.screen, color, 
                          (x - size//4, y - size//2, size//2, size//2), 
                          0, 3.14, 5)
    
    def _draw_clock(self):
        """Draw current time in top right"""
        now = datetime.now()
        time_str = now.strftime("%H:%M")
        time_text = self.font_small.render(time_str, True, self.WHITE)
        self.screen.blit(time_text, (self.width - 80, 10))
    
    # ----------------------------
    # Render Functions
    # ----------------------------
    def _render_loading(self):
        """Render loading screen"""
        self._draw_gradient_background(self.PRIMARY_DARK, self.PRIMARY)
        
        title_text = self.font_large.render("DOORBELL", True, self.WHITE)
        title_rect = title_text.get_rect(center=(self.width // 2, 80))
        self.screen.blit(title_text, title_rect)
        
        message = self.state_data.get("message", "Loading...")
        msg_text = self.font_small.render(message, True, self.PRIMARY_LIGHT)
        msg_rect = msg_text.get_rect(center=(self.width // 2, 180))
        self.screen.blit(msg_text, msg_rect)
        
        # Animated loading bar
        bar_width = 300
        bar_height = 8
        bar_x = (self.width - bar_width) // 2
        bar_y = 220
        
        pygame.draw.rect(self.screen, self.DARK_GRAY, 
                       (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        
        progress = (self.animation_time % 100) / 100
        progress_width = int(bar_width * progress)
        pygame.draw.rect(self.screen, self.WHITE, 
                       (bar_x, bar_y, progress_width, bar_height), border_radius=4)
    
    def _render_idle(self):
        """Render idle/ready state"""
        self.screen.fill(self.DARK_GRAY)
        self._draw_header("Smart Doorbell", datetime.now().strftime("%B %d, %Y"))
        self._draw_clock()
        
        center_y = 160
        door_locked = self.state_data.get("door_locked", True)
        
        self._draw_icon_lock(self.width // 2, center_y - 40, 60, locked=door_locked)
        
        status_text = "Door Locked" if door_locked else "Door Unlocked"
        status_color = self.RED if door_locked else self.GREEN
        text = self.font_medium.render(status_text, True, status_color)
        text_rect = text.get_rect(center=(self.width // 2, center_y + 30))
        self.screen.blit(text, text_rect)
        
        ready_text = self.font_small.render("Ready - Monitoring for faces", True, self.LIGHT_GRAY)
        ready_rect = ready_text.get_rect(center=(self.width // 2, center_y + 65))
        self.screen.blit(ready_text, ready_rect)
        
        self._call_button_rect = self._draw_button(
            self.width - 160, self.height - 60, 140, 45,
            "Call Owner", self.BLUE
        )
        
        self._draw_status_indicator(20, self.height - 30, self.GREEN, "Camera Active")
        self._draw_status_indicator(20, self.height - 55, self.GREEN, "System Online")
    
    def _render_detecting(self):
        """Render face detection screen"""
        self.screen.fill(self.DARK_GRAY)
        self._draw_header("Face Detected", "Analyzing...")
        self._draw_clock()
        
        center_y = 160
        pulse = abs((self.animation_time * 2) % 100 - 50) + 30
        pygame.draw.circle(self.screen, self.YELLOW, 
                         (self.width // 2, center_y), pulse, 3)
        
        text = self.font_medium.render("Identifying...", True, self.YELLOW)
        text_rect = text.get_rect(center=(self.width // 2, center_y + 60))
        self.screen.blit(text, text_rect)
        
        self._call_button_rect = self._draw_button(
            self.width - 160, self.height - 60, 140, 45,
            "Call Owner", self.BLUE
        )
    
    def _render_access_granted(self):
        """Render access granted screen"""
        self._draw_gradient_background(self.GREEN, self._darken_color(self.GREEN, 0.3))
        self._draw_header("Access Granted", "Welcome!")
        self._draw_clock()
        
        center_y = 160
        
        pygame.draw.circle(self.screen, self.WHITE, 
                         (self.width // 2, center_y - 20), 50, 6)
        points = [
            (self.width // 2 - 20, center_y - 20),
            (self.width // 2 - 5, center_y - 5),
            (self.width // 2 + 25, center_y - 45)
        ]
        pygame.draw.lines(self.screen, self.WHITE, False, points, 8)
        
        person_name = self.state_data.get("person_name", "User")
        name_text = self.font_large.render(person_name, True, self.WHITE)
        name_rect = name_text.get_rect(center=(self.width // 2, center_y + 50))
        self.screen.blit(name_text, name_rect)
        
        status_text = self.font_small.render("Door Unlocking...", True, self.WHITE)
        status_rect = status_text.get_rect(center=(self.width // 2, center_y + 85))
        self.screen.blit(status_text, status_rect)
    
    def _render_access_denied(self):
        """Render access denied screen"""
        self._draw_gradient_background(self.RED, self._darken_color(self.RED, 0.3))
        self._draw_header("Access Denied", "Unknown Person")
        self._draw_clock()
        
        center_y = 160
        
        pygame.draw.circle(self.screen, self.WHITE, 
                         (self.width // 2, center_y - 20), 50, 6)
        offset = 30
        pygame.draw.line(self.screen, self.WHITE,
                       (self.width // 2 - offset, center_y - 20 - offset),
                       (self.width // 2 + offset, center_y - 20 + offset), 8)
        pygame.draw.line(self.screen, self.WHITE,
                       (self.width // 2 + offset, center_y - 20 - offset),
                       (self.width // 2 - offset, center_y - 20 + offset), 8)
        
        warning_text = self.font_medium.render("Unknown Person", True, self.WHITE)
        warning_rect = warning_text.get_rect(center=(self.width // 2, center_y + 50))
        self.screen.blit(warning_text, warning_rect)
        
        status_text = self.font_small.render("Access Denied - Door Locked", True, self.WHITE)
        status_rect = status_text.get_rect(center=(self.width // 2, center_y + 85))
        self.screen.blit(status_text, status_rect)
        
        self._call_button_rect = self._draw_button(
            self.width // 2 - 70, self.height - 60, 140, 45,
            "Call Owner", self.WHITE, self.RED
        )
    
    def _render_calling(self):
        """Render calling screen"""
        self._draw_gradient_background(self.BLUE, self._darken_color(self.BLUE, 0.3))
        self._draw_header("Calling Owner", "Please wait...")
        self._draw_clock()
        
        center_y = 160
        pulse = abs((self.animation_time * 2) % 100 - 50) + 40
        pygame.draw.circle(self.screen, self.WHITE, 
                         (self.width // 2, center_y - 20), pulse, 5)
        
        # Draw phone icon using text
        phone_text = self.font_large.render("CALL", True, self.WHITE)
        phone_rect = phone_text.get_rect(center=(self.width // 2, center_y - 20))
        self.screen.blit(phone_text, phone_rect)
        
        status_text = self.font_medium.render("Connecting...", True, self.WHITE)
        status_rect = status_text.get_rect(center=(self.width // 2, center_y + 50))
        self.screen.blit(status_text, status_rect)
    
    def _render_status(self):
        """Render generic status screen"""
        status_type = self.state_data.get("status_type", "loading")
        message = self.state_data.get("message", "")
        
        color_map = {
            "loading": self.BLUE,
            "locked": self.RED,
            "unlocked": self.GREEN,
            "error": self.RED,
            "warning": self.YELLOW
        }
        
        bg_color = color_map.get(status_type, self.GRAY)
        
        self._draw_gradient_background(bg_color, self._darken_color(bg_color, 0.3))
        self._draw_header("Status", "")
        self._draw_clock()
        
        msg_text = self.font_medium.render(message, True, self.WHITE)
        msg_rect = msg_text.get_rect(center=(self.width // 2, 160))
        self.screen.blit(msg_text, msg_rect)