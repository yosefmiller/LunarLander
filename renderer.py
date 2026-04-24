import math
import os
from datetime import datetime

import imageio
import numpy as np
import pygame

from lunar_lander_env import LunarLanderEnv, SCREEN_WIDTH, SCREEN_HEIGHT, SCALE, GROUND_Y, PAD_WIDTH, LANDER_WIDTH, \
    LANDER_HEIGHT, FPS, SimpleLunarLanderEnv, LLE_XOffset, LLE_InitialVelocity, RandomLunarLander

# Additional graphics
NASA_LOGO = pygame.image.load('graphics/nasa_logo.png')
NASA_LOGO = pygame.transform.scale(NASA_LOGO, (30, 25))  # Resize
ASTRONAUT = pygame.image.load('graphics/astronaut.png')
ASTRONAUT = pygame.transform.scale(ASTRONAUT, (45, 50))  # Resize
AMERICAN_FLAG = pygame.image.load('graphics/american_flag.png')
AMERICAN_FLAG = pygame.transform.scale(AMERICAN_FLAG, (22, 15))  # Resize

# COLORS
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
GOLD = (212, 175, 55)  # Apollo Foil Gold
GREY = (100, 100, 100)
DARK_GREY = (50, 50, 50)
SUPER_DARK_GREY = (30, 30, 30)
BLUE = (50, 150, 255)
ORANGE = (255, 165, 0)
BEACON_BLUE = (0, 150, 255)
EARTH_BLUE = (30, 144, 255)
EARTH_GREEN = (34, 139, 34)
EARTH_CLOUD = (240, 240, 255)
SUN_YELLOW = (255, 255, 200)
MOON_LIGHT_GREY = (120, 120, 120)
MOON_DARK_GREY = (60, 60, 60)
ROCK_GREY = (90, 90, 90)

class Renderer:
    def __init__(self,
                 env: LunarLanderEnv,
                 agent_name='Agent',
                 save_video=False,
                 save_gif=False,
                 output_dir="recordings"):
        """
        Initialize Renderer

        Args:
            env: LunarLanderEnv instance
            agent_name: Name of agent to display in display caption
            save_video: If True, saves rendering as MP4 video
            save_gif: If True, saves rendering as GIF
            output_dir: Directory to save recordings
        """
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(f"Apollo Lunar Descent Simulation - {agent_name}")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 16)
        self.env = env
        self.env.record_trace = True

        # Recording setup
        self.save_video = save_video
        self.save_gif = save_gif
        self.output_dir = output_dir
        self.frames = []
        self.recording_active = False
        self.episode_count = 0

        if save_video or save_gif:
            os.makedirs(output_dir, exist_ok=True)
            self.recording_active = True

        # Beacon animation
        self.beacon_timer = 0

        # Generate terrain features (boulders, rocks, craters)
        self._generate_terrain_features()

        # Earth position in sky (upper right)
        self.earth_pos = (SCREEN_WIDTH - 120, 80)
        self.earth_radius = 40

    def _generate_terrain_features(self):
        """Generate random boulders, rocks, and craters for the moon surface."""
        rng = np.random.default_rng(42) # For consistent terrain across renders

        # Mountains
        self.mountains = [
            # mountain 1
            [(150, SCREEN_HEIGHT - 130),  # Top-left
            (250, SCREEN_HEIGHT - 130),  # Top-right
            (300, SCREEN_HEIGHT - 100),  # Bottom-right
            (100, SCREEN_HEIGHT - 100)   # Bottom-left
            ],
            # mountain 2
            [(SCREEN_WIDTH - 350, SCREEN_HEIGHT - 120),  # Top-left
            (SCREEN_WIDTH - 400, SCREEN_HEIGHT - 130),  # Peak
            (SCREEN_WIDTH - 450, SCREEN_HEIGHT - 120),  # Top-right
            (SCREEN_WIDTH - 550, SCREEN_HEIGHT - 90),  # Bottom-right
            (SCREEN_WIDTH - 250, SCREEN_HEIGHT - 90)   # Bottom-left
            ],
            # mountain 3
            [(SCREEN_WIDTH - 50, SCREEN_HEIGHT - 110),  # Top-left
            (SCREEN_WIDTH - 10, SCREEN_HEIGHT - 120),  # Peak
            (SCREEN_WIDTH, SCREEN_HEIGHT - 110),  # Top-right
            (SCREEN_WIDTH, SCREEN_HEIGHT - 80),  # Bottom-right
            (SCREEN_WIDTH - 100, SCREEN_HEIGHT - 80)   # Bottom-left
            ]
        ]

        # Craters
        self.craters = [
            # crater 1
            [(SCREEN_WIDTH - 250, SCREEN_HEIGHT - 60),  # Top-left
             (SCREEN_WIDTH - 150, SCREEN_HEIGHT - 60),  # Top-right
             (SCREEN_WIDTH - 100, SCREEN_HEIGHT - 70),  # Bottom-right
             (SCREEN_WIDTH - 300, SCREEN_HEIGHT - 70)   # Bottom-left
            ],
            # crater 2
            [(300, SCREEN_HEIGHT - 70),  # Top-left
             (200, SCREEN_HEIGHT - 70),  # Top-right
             (150, SCREEN_HEIGHT - 80),  # Bottom-right
             (350, SCREEN_HEIGHT - 80)   # Bottom-left
            ]
        ]

        # Boulders
        self.boulders = []
        for _ in range(6):
            x = rng.uniform(0, SCREEN_WIDTH)
            _, middle_ground = self.world_to_screen(0, 0)
            offset = rng.uniform(-25, 25)
            y = middle_ground + offset
            size = rng.uniform(8, 25)
            num_sides = rng.integers(5, 8)
            self.boulders.append((x, y, size, num_sides))

        # Small rocks
        self.rocks = []
        for _ in range(12):
            x = rng.uniform(0, SCREEN_WIDTH)
            _, middle_ground = self.world_to_screen(0, 0)
            offset = rng.uniform(-25, 25)
            y = middle_ground + offset
            size = rng.uniform(2, 8)
            self.rocks.append((x, y, size))

    @staticmethod
    def world_to_screen(x, y):
        # Center x on screen (Screen Center = World 0)
        screen_x = int(SCREEN_WIDTH / 2 + x * SCALE)
        # Flip Y (Screen 0 is top)
        screen_y = int(SCREEN_HEIGHT - (GROUND_Y + y * SCALE))
        return screen_x, screen_y

    def draw_earth(self):
        """Draw Earth in the background with partial shadow from the Sun."""
        cx, cy = self.earth_pos
        radius = self.earth_radius

        # Draw Earth base (blue)
        pygame.draw.circle(self.screen, EARTH_BLUE, (cx, cy), radius)

        # Add some green "continents" with irregular shapes
        pygame.draw.circle(self.screen, EARTH_GREEN, (cx - 15, cy - 10), 12)
        pygame.draw.circle(self.screen, EARTH_GREEN, (cx + 10, cy + 8), 15)
        pygame.draw.circle(self.screen, EARTH_GREEN, (cx - 5, cy + 15), 8)
        pygame.draw.circle(self.screen, EARTH_GREEN, (cx + 18, cy - 5), 10)

        # Add white clouds
        cloud_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(cloud_surface, (*EARTH_CLOUD, 100), (radius - 10, radius + 5), 8)
        pygame.draw.circle(cloud_surface, (*EARTH_CLOUD, 120), (radius + 12, radius - 8), 6)
        self.screen.blit(cloud_surface, (cx - radius, cy - radius))

        # Draw shadow from the Sun (Sun is on the left side, so shadow is on right)
        # Create a darker crescent on the right side
        shadow_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        # Shadow circle shifted to create crescent effect
        pygame.draw.circle(shadow_surface, (0, 0, 0, 120), (radius + 18, radius), radius)
        self.screen.blit(shadow_surface, (cx - radius, cy - radius))

        # Add atmospheric glow
        glow_surface = pygame.Surface((radius * 3, radius * 3), pygame.SRCALPHA)
        for i in range(3):
            alpha = 30 - i * 8
            pygame.draw.circle(glow_surface, (*EARTH_BLUE, alpha),
                             (int(radius * 1.5), int(radius * 1.5)),
                             radius + i * 5, 2)
        self.screen.blit(glow_surface, (cx - int(radius * 1.5), cy - int(radius * 1.5)))

    def draw_terrain_features(self, ground_px):
        """Draw mountains, craters, boulders, and rocks on the moon surface."""

        # Draw horizon mountains
        pygame.draw.polygon(self.screen, MOON_DARK_GREY, self.mountains[0])
        pygame.draw.polygon(self.screen, MOON_DARK_GREY, self.mountains[1])
        pygame.draw.polygon(self.screen, MOON_DARK_GREY, self.mountains[2])

        # Draw craters
        pygame.draw.polygon(self.screen, MOON_DARK_GREY, self.craters[0])
        pygame.draw.polygon(self.screen, MOON_DARK_GREY, self.craters[1])

        # Draw boulders
        for boulder_x, boulder_y, boulder_size, num_sides in self.boulders:
            # Draw boulder as irregular polygon
            pts = []
            angles = np.linspace(0, 2*np.pi, num_sides, endpoint=False)
            for i, angle in enumerate(angles):
                # Add irregularity
                r = boulder_size * (0.7 + (i % 3) * 0.2)
                px = boulder_x + r * np.cos(angle)
                py = boulder_y - boulder_size/2 + r * np.sin(angle) * 0.7
                pts.append((int(px), int(py)))
            pygame.draw.polygon(self.screen, ROCK_GREY, pts)
            pygame.draw.polygon(self.screen, (70, 70, 70), pts, 2)

            # Add highlight for 3D effect
            highlight_pts = [(pts[0][0], pts[0][1] - 2),
                           (pts[1][0], pts[1][1] - 1)]
            if len(highlight_pts) == 2:
                pygame.draw.line(self.screen, MOON_LIGHT_GREY,
                               highlight_pts[0], highlight_pts[1], 2)

        # Draw small rocks
        for rock_x, rock_y, rock_size in self.rocks:
            pygame.draw.circle(self.screen, (80, 80, 80),
                             (int(rock_x), int(rock_y - rock_size/2)),
                             int(rock_size))
            # Add small highlight
            pygame.draw.circle(self.screen, MOON_LIGHT_GREY,
                             (int(rock_x - rock_size/4), int(rock_y - rock_size/2 - rock_size/4)),
                             int(rock_size/3))

    def draw_landing_pad(self):
        """Draw the landing pad."""
        pad_center = self.env.pad_x_offset
        pad_left = self.env.pad_x_offset - PAD_WIDTH / 2
        pad_right = self.env.pad_x_offset + PAD_WIDTH / 2
        pad_center_px, pad_center_py = self.world_to_screen(pad_center, 0)

        # Pad surface
        pygame.draw.ellipse(self.screen, SUPER_DARK_GREY, (pad_center_px - 100, SCREEN_HEIGHT - 50, 200, 20))
        pygame.draw.ellipse(self.screen, WHITE, (pad_center_px - 50, SCREEN_HEIGHT - 47, 100, 10), 2)

        # Pad markers
        for x in [pad_left, pad_right]:
            px = self.world_to_screen(x, -1)
            pygame.draw.line(self.screen, DARK_GREY, px,
                           (px[0], px[1] - 20), 3)

        # Landing marker circle
        pygame.draw.circle(self.screen, WHITE, (int(pad_center_px), pad_center_py - 3), 5)
        pygame.draw.circle(self.screen, DARK_GREY, (int(pad_center_px), pad_center_py - 3), 3)

        # Draw H marking for helipad
        h_font = pygame.font.SysFont("monospace", 24, bold=True)
        h_text = h_font.render("H", True, WHITE)
        h_rect = h_text.get_rect(center=(int(pad_center_px), pad_center_py - 18))
        self.screen.blit(h_text, h_rect)

    def draw_beacons(self, ground_px):
        """Draw blinking blue beacons 5 meters on either side of landing pad."""
        # Update beacon timer
        self.beacon_timer += 1

        # Beacons blink with 2-second period (120 frames at 60 FPS)
        blink_on = (self.beacon_timer % 120) < 60

        if blink_on:
            # Left beacon (5 meters from left edge of pad)
            left_beacon_world_x = self.env.pad_x_offset - PAD_WIDTH/2
            left_beacon_x, _ = self.world_to_screen(left_beacon_world_x, 0)
            _, left_beacon_y = self.world_to_screen(left_beacon_world_x, 1)

            # Right beacon (5 meters from right edge of pad)
            right_beacon_world_x = self.env.pad_x_offset + PAD_WIDTH/2
            right_beacon_x, _ = self.world_to_screen(right_beacon_world_x, 0)
            _, right_beacon_y = self.world_to_screen(right_beacon_world_x, 1)

            # Draw beacons with glow effect
            for beacon_x, beacon_y in [(left_beacon_x, left_beacon_y),
                                       (right_beacon_x, right_beacon_y)]:
                # Outer glow (largest)
                glow_surf = pygame.Surface((30, 30), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (*BEACON_BLUE, 30), (15, 15), 15)
                self.screen.blit(glow_surf, (int(beacon_x - 15), int(beacon_y - 15)))

                # Middle glow
                pygame.draw.circle(self.screen, (*BEACON_BLUE, 150),
                                 (int(beacon_x), int(beacon_y)), 8)
                # Bright core
                pygame.draw.circle(self.screen, BEACON_BLUE,
                                 (int(beacon_x), int(beacon_y)), 5)
                # Very bright center
                pygame.draw.circle(self.screen, WHITE,
                                 (int(beacon_x), int(beacon_y)), 2)

    def draw_bins(self, bins: dict):
        # X grid lines
        for bx in bins.get('x', []):
            wx = bx * 50.0  # Convert normalized x back to world
            sx1, sy1 = self.world_to_screen(wx, -100)  # Extend from y=-100 to y=100
            sx2, sy2 = self.world_to_screen(wx, 100)
            pygame.draw.line(self.screen, GREY, (sx1, sy1), (sx2, sy2), 1)

        # Y grid lines
        for by in bins.get('y', []):
            wy = by * 50.0  # Convert normalized y back to world
            sx1, sy1 = self.world_to_screen(-100, wy)  # Extend from x=-100 to x=100
            sx2, sy2 = self.world_to_screen(100, wy)
            pygame.draw.line(self.screen, GREY, (sx1, sy1), (sx2, sy2), 1)

        # Theta radial grid centered on lander
        lander_sx, lander_sy = self.world_to_screen(self.env.x, self.env.y)
        for btheta in bins.get('theta', []):
            # Draw radial lines at bin angles, length 100 pixels
            dx = math.sin(btheta) * 100
            dy = math.cos(btheta) * 100
            end_sx = lander_sx + dx
            end_sy = lander_sy - dy  # Screen y is flipped
            pygame.draw.line(self.screen, GREY, (lander_sx, lander_sy), (end_sx, end_sy), 1)

    def render(self, action=0, bins=None):
        self.screen.fill(BLACK)

        # 1. Earth in the background
        self.draw_earth()

        # 2. Stars (Static background decoration)
        rng = np.random.default_rng(12345) # Consistent stars across frames
        for i in range(150):
            rng.uniform()
            sx = int(rng.uniform() * SCREEN_WIDTH)
            sy = int(rng.uniform() * SCREEN_HEIGHT * 0.80)  # Only in upper 80%
            brightness = int(150 + rng.uniform() * 105)
            size = 1 if rng.uniform() > 0.7 else 0
            if size == 0:
                self.screen.set_at((sx, sy), (brightness, brightness, brightness))
            else:
                pygame.draw.circle(self.screen, (brightness, brightness, brightness), (sx, sy), 1)

        # 3. Moon Surface
        # Add depth beyond landing pad
        ground_px = int(SCREEN_HEIGHT - GROUND_Y - 50)
        pygame.draw.rect(self.screen, MOON_LIGHT_GREY, (0, ground_px, SCREEN_WIDTH, SCREEN_HEIGHT - ground_px))
        ground_px = int(SCREEN_HEIGHT - GROUND_Y)
        pygame.draw.rect(self.screen, MOON_LIGHT_GREY, (0, ground_px, SCREEN_WIDTH, SCREEN_HEIGHT - ground_px))

        # 4. Draw terrain features (craters, boulders, rocks)
        self.draw_terrain_features(ground_px)

        # 5. Landing Pad (Target)
        self.draw_landing_pad()

        # 6. Draw blinking beacons
        self.draw_beacons(ground_px)

        # 7. Draw astronaut and flag
        self.screen.blit(ASTRONAUT, (325, SCREEN_HEIGHT - 95))
        self.screen.blit(AMERICAN_FLAG, (377, SCREEN_HEIGHT - 95))
        pygame.draw.rect(self.screen, WHITE, (376, SCREEN_HEIGHT - 95, 1, 30))

        if bins is not None:
            self.draw_bins(bins)

        # 8. Trajectory Trace
        if len(self.env.trace) > 1:
            pts = [self.world_to_screen(px, py) for px, py in self.env.trace]
            # Draw with gradient effect (older = more transparent)
            for i in range(1, len(pts)):
                alpha = int(100 + (i / len(pts)) * 155)
                color = (*BLUE[:2], min(255, BLUE[2] + 50))
                pygame.draw.line(self.screen, color, pts[i-1], pts[i], 1)

        # 9. Draw Lander (LEM Style)
        w = LANDER_WIDTH * SCALE
        h = LANDER_HEIGHT * SCALE

        surf_size = int(max(w, h) * 3)
        lander_surf = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
        cx, cy = surf_size // 2, surf_size // 2

        # -- DESCENT STAGE (Gold Octagon-ish) --
        pygame.draw.rect(lander_surf, GOLD, (cx - w / 2, cy, w, h / 1.5))
        lander_surf.blit(NASA_LOGO, (cx - w / 2 + 15, cy + 1))
        # Legs
        pygame.draw.line(lander_surf, GOLD, (cx - w / 2, cy + h / 2), (cx - w / 2 - 10, cy + h), 3)
        pygame.draw.line(lander_surf, GOLD, (cx + w / 2, cy + h / 2), (cx + w / 2 + 10, cy + h), 3)
        # Footpads
        pygame.draw.circle(lander_surf, GREY, (int(cx - w / 2 - 10), int(cy + h)), 5)
        pygame.draw.circle(lander_surf, GREY, (int(cx + w / 2 + 10), int(cy + h)), 5)

        # -- ASCENT STAGE (Grey/Black Top) --
        pygame.draw.rect(lander_surf, MOON_DARK_GREY, (cx - w / 2.2, cy - h / 2, w / 1.1, h / 2))
        pygame.draw.polygon(lander_surf, GREY, [(cx - w / 2.2, cy - h / 2), (cx, cy - h), (cx + w / 2.2, cy - h / 2)])

        # -- FLAMES --
        if self.env.fuel > 0:
            if action == 1:  # Main Engine
                flame_pts = [(cx - w / 3, cy + h / 1.5),
                            (cx + w / 3, cy + h / 1.5),
                            (cx, cy + h * 1.5)]
                pygame.draw.polygon(lander_surf, ORANGE, flame_pts)
                pygame.draw.polygon(lander_surf, RED,
                                  [(cx - w / 4, cy + h / 1.5),
                                   (cx + w / 4, cy + h / 1.5),
                                   (cx, cy + h * 1.3)])
            if action == 2:  # Left RCS
                pygame.draw.circle(lander_surf, ORANGE, (int(cx + w / 2), int(cy - h / 4)), 4)
                pygame.draw.circle(lander_surf, WHITE, (int(cx + w / 2), int(cy - h / 4)), 2)
            if action == 3:  # Right RCS
                pygame.draw.circle(lander_surf, ORANGE, (int(cx - w / 2), int(cy - h / 4)), 4)
                pygame.draw.circle(lander_surf, WHITE, (int(cx - w / 2), int(cy - h / 4)), 2)

        # Rotate and Blit
        rot_surf = pygame.transform.rotate(lander_surf, math.degrees(self.env.theta))
        rect = rot_surf.get_rect()
        rect.center = self.world_to_screen(self.env.x, self.env.y)
        self.screen.blit(rot_surf, rect)

        # 10. HUD / Telemetry
        telemetry = [
            f"ALTITUDE: {self.env.y:.1f} m",
            f"H-SPEED:  {self.env.vx:.1f} m/s",
            f"V-SPEED:  {self.env.vy:.1f} m/s",
            f"ANGLE:    {math.degrees(self.env.theta):.1f} deg",
            f"ANG.VEL:  {math.degrees(self.env.omega):.1f} deg/s",
            f"FUEL:     {self.env.fuel:.1f} kg",
            f"MASS:     {self.env.mass:.0f} kg",
            f"SHAPING:  {self.env.calculate_shaping():.2f}",
        ]

        for i, line in enumerate(telemetry):
            color = WHITE
            if "FUEL" in line and self.env.fuel < 50:
                color = RED
            if "V-SPEED" in line and abs(self.env.vy) > 2.5:
                color = RED
            if "ANGLE" in line and abs(self.env.theta) > 0.3:
                color = RED
            txt = self.font.render(line, True, color)
            self.screen.blit(txt, (10, 10 + i * 20))

        # 11. Status Messages
        if self.env.landed:
            s = pygame.Surface((SCREEN_WIDTH, 60), pygame.SRCALPHA)
            s.fill((0, 255, 0, 100))
            self.screen.blit(s, (0, SCREEN_HEIGHT / 2 - 30))
            msg_text = "TOUCHDOWN CONFIRMED - R to Reset"
            msg = self.font.render(msg_text, True, WHITE)
            msg_rect = msg.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 - 10))
            self.screen.blit(msg, msg_rect)

        if self.env.crashed or self.env.timeout:
            s = pygame.Surface((SCREEN_WIDTH, 60), pygame.SRCALPHA)
            s.fill((255, 0, 0, 100))
            self.screen.blit(s, (0, SCREEN_HEIGHT / 2 - 30))
            crash_msg = f"VEHICLE CRASHED: {self.env.crash_reason} - R to Reset" if self.env.crashed else "VEHICLE TIMEOUT - R to Reset"
            msg = self.font.render(crash_msg, True, WHITE)
            self.screen.blit(msg, (SCREEN_WIDTH / 2 - 200, SCREEN_HEIGHT / 2 - 10))

        pygame.display.flip()

        # Capture frame for recording
        if self.recording_active:
            # Convert pygame surface to numpy array
            frame = pygame.surfarray.array3d(self.screen)
            frame = np.transpose(frame, (1, 0, 2))  # Pygame uses (width, height, channels)
            self.frames.append(frame.copy())

    def change_env(self, new_env: LunarLanderEnv):
        """Change the environment being rendered."""
        self.env = new_env
        self.env.record_trace = True
        self.frames = []

    def start_recording(self):
        """Start a new recording session."""
        self.frames = []
        self.recording_active = True

    def stop_recording(self):
        """Stop recording without saving."""
        self.recording_active = False
        self.frames = []

    def save_recording(self, filename=None):
        """Save recorded frames as video or GIF."""
        if not self.frames:
            print("No frames to save!")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lunar_landing_{timestamp}"

        # Remove extension if provided
        filename = os.path.splitext(filename)[0]

        try:
            if self.save_video:
                video_path = os.path.join(self.output_dir, f"{filename}.mp4")
                print(f"Saving video to {video_path}...")
                imageio.mimsave(video_path, self.frames, fps=FPS, codec='libx264',
                              quality=8, pixelformat='yuv420p')
                print(f"Video saved successfully! ({len(self.frames)} frames)")

            if self.save_gif:
                gif_path = os.path.join(self.output_dir, f"{filename}.gif")
                print(f"Saving GIF to {gif_path}...")
                # Reduce frame rate for GIF to reduce file size
                gif_frames = self.frames[::2] + [self.frames[-1]] # Every other frame
                imageio.mimsave(gif_path, gif_frames, fps=FPS//2, loop=0)
                print(f"GIF saved successfully! ({len(gif_frames)} frames)")

        except Exception as e:
            print(f"Error saving recording: {e}")
        finally:
            self.frames = []
            self.episode_count += 1

def main():
    env = LunarLanderEnv()
    # env = SimpleLunarLanderEnv()
    renderer = Renderer(
        env,
        save_video=False,  # Set to True to save as MP4
        save_gif=True,  # Set to True to save as GIF
        output_dir="lunar_recordings"
    )
    running = True
    done = False

    print("=" * 60)
    print("APOLLO LUNAR DESCENT SIMULATION")
    print("=" * 60)
    print("Controls:")
    print("  UP ARROW    - Main Engine (Thrust)")
    print("  LEFT ARROW  - Left RCS (Rotate Clockwise)")
    print("  RIGHT ARROW - Right RCS (Rotate Counter-Clockwise)")
    print("  R           - Reset Environment")
    print("  S           - Save Recording (if recording enabled)")
    print("  Q / ESC     - Quit")
    print("=" * 60)
    print(f"Landing Pad Offset: {env.pad_x_offset:.1f} meters")
    print("=" * 60)

    while running:
        renderer.clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # Save current episode before reset
                    # if renderer.recording_active and len(renderer.frames) > 0:
                    #     renderer.save_recording()
                    env.reset()
                    done = False

                if event.key == pygame.K_s:
                    # Manual save
                    if len(renderer.frames) > 0:
                        renderer.save_recording()
                    else:
                        print("No frames to save!")

                if event.key in [pygame.K_q, pygame.K_ESCAPE]:
                    running = False

        # Get keyboard input for actions
        keys = pygame.key.get_pressed()
        action = 0
        if keys[pygame.K_UP]:
            action = 1
        elif keys[pygame.K_LEFT]:
            action = 2
        elif keys[pygame.K_RIGHT]:
            action = 3

        if keys[pygame.K_1]:
            env = SimpleLunarLanderEnv()
            renderer.change_env(env)
            done = False
            continue
        elif keys[pygame.K_2]:
            env = LLE_XOffset()
            renderer.change_env(env)
            done = False
            continue
        elif keys[pygame.K_3]:
            env = LLE_InitialVelocity()
            renderer.change_env(env)
            done = False
            continue
        elif keys[pygame.K_4]:
            env = LunarLanderEnv()
            renderer.change_env(env)
            done = False
            continue
        elif keys[pygame.K_5]:
            env = RandomLunarLander()
            renderer.change_env(env)
            done = False
            continue

        if keys[pygame.K_q]:
            break

        if not done:
            state, reward, done, result = env.step(action)

            # Auto-save recording when episode ends
            if done and renderer.recording_active:
                status = "landed" if result['landed'] else ("crashed" if result['crashed'] else "timeout")
                filename = f"episode_{renderer.episode_count}_{status}"
                # renderer.save_recording(filename)

        renderer.render(action)

    pygame.quit()
    print("\nSimulation ended. Goodbye!")

if __name__ == "__main__":
    main()