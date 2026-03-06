from dataclasses import dataclass

import numpy as np
import pygame
import math

# --- Constants ---
FPS = 60
DT = 1.0 / FPS

# PHYSICS CONSTANTS (MOON)
GRAVITY = -1.625  # Moon gravity (approx 1/6 Earth)
SCALE = 10.0  # Pixels per meter (Zoomed out to show approach)

# WORLD
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
GROUND_Y = 50.0  # Meters from bottom of screen (visual offset)

# LANDER PROPERTIES (Apollo-esque ratios)
# We separate dry mass and fuel mass to simulate variable acceleration
LANDER_DRY_MASS = 600.0  # kg (Structure + Descent Engine)
FUEL_MASS_START = 400.0  # kg (Propellant)
MAX_FUEL = 400.0

LANDER_WIDTH = 6.0  # Wider stance like the LEM
LANDER_HEIGHT = 4.0

# THRUST PARAMETERS
# Weight (Full) = 1000kg * 1.625 = 1625 N
# Max Thrust should be ~2-3x Weight for control.
# Apollo Descent Engine was throttlable, here we simulate 'max' burst.
MAIN_THRUST = 4500.0  # T/W Ratio ~ 2.7 (at start) -> ~ 4.6 (when empty)
SIDE_THRUST = 1000.0  # RCS Thrusters
SIDE_ENGINE_OFFSET = 3.0  # Torque leverage

FUEL_CONSUMPTION_MAIN = 2.0  # kg per second (approx)
FUEL_CONSUMPTION_SIDE = 0.5

# LANDING ZONE
PAD_WIDTH = 20.0  # Meters
PAD_X_TARGET = 0.0  # Center of the world (0,0)

# COLORS
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
GOLD = (212, 175, 55)  # Apollo Foil Gold
GREY = (100, 100, 100)
DARK_GREY = (50, 50, 50)
BLUE = (50, 150, 255)
ORANGE = (255, 165, 0)


@dataclass
class LunarLanderState:
    """Represents the state of the lunar lander."""
    x: float  # Relative distance X (normalized)
    y: float  # Relative distance Y (normalized)
    vx: float  # Velocity X (normalized)
    vy: float  # Velocity Y (normalized)
    theta: float  # Angle (radians)
    omega: float  # Angular velocity
    fuel: float  # Fuel remaining (normalized)
    on_pad: float  # Pad contact sensor (0.0 or 1.0)

    def to_array(self) -> np.ndarray:
        """Convert state to numpy array for RL algorithms."""
        return np.array([
            self.x, self.y, self.vx, self.vy,
            self.theta, self.omega, self.fuel, self.on_pad
        ], dtype=np.float32)


class LunarLanderEnv:
    x: float
    y: float
    vx: float
    vy: float
    theta: float
    omega: float
    fuel: float
    mass: float
    landed: bool
    crashed: bool
    crash_reason: str
    trace: list
    prev_shaping: float|None
    moment_of_inertia: float


    def __init__(self):
        self.reset()

    def reset(self) -> LunarLanderState:
        self._init_state()

        self.fuel = FUEL_MASS_START
        self.mass = LANDER_DRY_MASS + self.fuel

        self.landed = False
        self.crashed = False
        self.crash_reason = ""  # Store reason for display
        self.trace = []

        self.prev_shaping = None

        # Approximate Moment of Inertia (Rectangle)
        # Note: In reality, `I` changes as fuel burns, but we keep `I` constant for stability
        self.moment_of_inertia = LANDER_DRY_MASS * (LANDER_WIDTH ** 2 + LANDER_HEIGHT ** 2) / 12.0

        return self._get_state()

    def _init_state(self):
        # 1. INITIALIZATION (Apollo "High Gate" Style)
        # Instead of dropping straight down, we come in with lateral speed.
        # This requires a "Braking Burn".

        start_x_range = [-60, -40] if np.random.rand() > 0.5 else [40, 60]
        self.x = np.random.uniform(start_x_range[0], start_x_range[1])
        self.y = np.random.uniform(40, 50)  # Start high up

        # Velocity points towards the center, but fast
        direction = -1.0 if self.x > 0 else 1.0
        self.vx = np.random.uniform(5.0, 10.0) * direction
        self.vy = np.random.uniform(-2.0, 0.0)  # Slight downward drift

        # Random initial tilt (imperfection)
        self.theta = np.random.uniform(-0.2, 0.2)
        self.omega = 0.0

    def _calculate_shaping(self) -> float:
        # Potential-based reward shaping
        # Goal: (0,0) position, 0 velocity, 0 angle
        dist_penalty = np.sqrt(self.x ** 2 + self.y ** 2)
        vel_penalty = np.sqrt(self.vx ** 2 + self.vy ** 2)
        tilt_penalty = abs(self.theta)

        # Weights
        return -10.0 * dist_penalty - 100.0 * vel_penalty - 100.0 * tilt_penalty

    def step(self, action):
        """
        Action space:
        0: Do nothing
        1: Left Engine (Rotates CW)
        2: Main Engine
        3: Right Engine (Rotates CCW)
        """
        if self.landed or self.crashed:
            return self._get_state(), 0, True, {}

        # Update Mass based on fuel burn
        self.mass = LANDER_DRY_MASS + self.fuel

        force_x = 0.0
        force_y = self.mass * GRAVITY  # Weight
        torque = 0.0

        # Apply Thrust
        # if self.fuel > 0:
        sin_theta = np.sin(self.theta)
        cos_theta = np.cos(self.theta)

        match action:
            case 0:  # No Action
                pass
            case 1:  # Main Engine
                f_thrust = MAIN_THRUST
                force_x += -sin_theta * f_thrust
                force_y += cos_theta * f_thrust
                self.fuel -= FUEL_CONSUMPTION_MAIN * DT
            case 2:  # Left RCS
                # Standard RL simplified: "Left" action fires left thruster, pushes ship RIGHT, rotates CW
                f_thrust = SIDE_THRUST
                force_x += cos_theta * f_thrust
                force_y += sin_theta * f_thrust
                torque -= f_thrust * SIDE_ENGINE_OFFSET
                self.fuel -= FUEL_CONSUMPTION_SIDE * DT
            case 3:  # Right RCS
                f_thrust = SIDE_THRUST
                force_x -= cos_theta * f_thrust
                force_y -= sin_theta * f_thrust
                torque += f_thrust * SIDE_ENGINE_OFFSET
                self.fuel -= FUEL_CONSUMPTION_SIDE * DT
            case _:
                raise ValueError("Invalid action")

        # Integrate (Newton's Second Law: F = ma)
        accel_x = force_x / self.mass
        accel_y = force_y / self.mass

        self.vx += accel_x * DT
        self.vy += accel_y * DT
        self.x += self.vx * DT
        self.y += self.vy * DT

        alpha = torque / self.moment_of_inertia
        self.omega += alpha * DT
        self.theta += self.omega * DT

        self.trace.append((self.x, self.y))
        if len(self.trace) > 200: self.trace.pop(0)

        # Reward Calculation
        shaping = self._calculate_shaping()
        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # Constant fuel penalty (encourages time-optimality)
        reward -= 0.05

        # Collision Check
        done = False

        # Legs: Define the LEM feet relative to center
        # LEM had wide landing gear
        half_w = LANDER_WIDTH / 2.0 + 1.0  # Gear extends past body
        half_h = LANDER_HEIGHT / 2.0 + 1.0  # Gear extends below body

        def get_world_point(lx, ly):
            c = np.cos(self.theta)
            s = np.sin(self.theta)
            return self.x + lx * c - ly * s, self.y + lx * s + ly * c

        # Check Left and Right foot
        left_foot = get_world_point(-half_w, -half_h)
        right_foot = get_world_point(half_w, -half_h)

        # Ground Plane (y=0)
        if left_foot[1] <= 0 or right_foot[1] <= 0:
            done = True

            # Landing Criteria (Apollo style: very gentle)
            # Horizontal speed must be near zero
            # Vertical speed must be < 2.5 m/s
            # Angle must be level
            # Must be within PAD radius

            vel_safe = abs(self.vy) < 2.5 and abs(self.vx) < 2.0
            angle_safe = abs(self.theta) < 0.3
            on_pad = abs(self.x) < (PAD_WIDTH / 2.0)

            # Physics adjustment to sit on ground
            min_y = min(left_foot[1], right_foot[1])
            self.y -= min_y

            if vel_safe and angle_safe and on_pad:
                self.landed = True
                reward += 100.0
                print(f"EAGLE HAS LANDED. Fuel Left: {self.fuel:.1f}kg")
            else:
                self.crashed = True
                reward -= 100.0
                reason = []
                if not vel_safe: reason.append("Too Fast")
                if not angle_safe: reason.append("Tilted")
                if not on_pad: reason.append("Missed LZ")
                self.crash_reason = ", ".join(reason)
                print(f"ABORT/CRASH: {self.crash_reason}")

        # Out of bounds
        if abs(self.x) > (SCREEN_WIDTH / SCALE) / 2 + 20 or self.y > (SCREEN_HEIGHT / SCALE) + 20:
            done = True
            self.crashed = True
            self.crash_reason = "Out of Bounds"
            reward -= 100.0

        return self._get_state(), reward, done, {}

    def _get_state(self) -> LunarLanderState:
        # RL CRITICAL UPDATE: RELATIVE NAVIGATION
        # We output position relative to the target (0,0), not absolute screen coords.
        # This allows the agent to generalize to different landing pad locations.

        return LunarLanderState(
            x=self.x / 50.0,  # Relative Dist X (Normalized)
            y=self.y / 50.0,  # Relative Dist Y
            vx=self.vx / 10.0,  # Vel X
            vy=self.vy / 10.0,  # Vel Y
            theta=self.theta,  # Angle
            omega=self.omega,  # Angular Vel
            fuel=self.fuel / FUEL_MASS_START,
            on_pad=1.0 if (abs(self.x) < PAD_WIDTH / 2) else 0.0  # Pad Contact Sensor
        )


class SimpleLunarLanderEnv(LunarLanderEnv):
    """
    Simplified lunar lander environment: drops straight down (no lateral speed, no tilt).
    Inherits from LunarLanderEnv to avoid code duplication.

    Key differences:
    - Starts at x=0 (centered on landing pad) instead of random offset
    - No lateral velocity (vx=0) instead of 5-10 m/s
    - No initial tilt (theta=0) instead of random ±0.2 rad
    - Randomized downward velocity (vy: -3.0 to -0.5 m/s)
    """

    def _init_state(self):
        """Override _init_sate() to provide simpler initial conditions."""
        # Simple initialization: straight drop
        self.x = 0.0  # Centered on landing pad
        self.y = np.random.uniform(40, 50)  # Start high up

        # No lateral velocity
        self.vx = 0.0
        # Randomized downward velocity (steeper than complex env)
        self.vy = np.random.uniform(-3.0, -0.5)

        # No initial tilt
        self.theta = 0.0
        self.omega = 0.0

    def step(self, action):
        if action not in [0, 1]:
            raise ValueError("Invalid action for SimpleLunarLanderEnv. Only 0 (No Action) and 1 (Main Engine) are allowed.")
        return super().step(action)

class Renderer:
    def __init__(self, env):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Apollo Lunar Descent Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 16)
        self.env = env

    def world_to_screen(self, x, y):
        # Center x on screen (Screen Center = World 0)
        screen_x = int(SCREEN_WIDTH / 2 + x * SCALE)
        # Flip Y (Screen 0 is top)
        screen_y = int(SCREEN_HEIGHT - (GROUND_Y + y * SCALE))
        return screen_x, screen_y

    def render(self, action=0):
        self.screen.fill(BLACK)

        # 1. Stars (Static background decoration)
        for i in range(50):
            sx = (i * 137) % SCREEN_WIDTH
            sy = (i * 93) % SCREEN_HEIGHT
            self.screen.set_at((sx, sy), WHITE)

        # 2. Moon Surface
        ground_px = int(SCREEN_HEIGHT - GROUND_Y)
        pygame.draw.rect(self.screen, GREY, (0, ground_px, SCREEN_WIDTH, SCREEN_HEIGHT - ground_px))

        # 3. Landing Pad (Target)
        pad_w_px = PAD_WIDTH * SCALE
        pad_left, _ = self.world_to_screen(-PAD_WIDTH / 2, 0)
        pygame.draw.rect(self.screen, DARK_GREY, (pad_left, ground_px, pad_w_px, 10))
        # Landing marker
        pygame.draw.circle(self.screen, WHITE, (int(SCREEN_WIDTH / 2), ground_px + 5), 5)

        # 4. Trajectory Trace
        if len(self.env.trace) > 1:
            pts = [self.world_to_screen(px, py) for px, py in self.env.trace]
            pygame.draw.lines(self.screen, BLUE, False, pts, 1)

        # 5. Draw Lander (LEM Style)
        # We draw to a surface then rotate
        # LEM Dimensions scaled
        w = LANDER_WIDTH * SCALE
        h = LANDER_HEIGHT * SCALE

        # Bigger surface to accommodate rotation and legs
        surf_size = int(max(w, h) * 3)
        lander_surf = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
        cx, cy = surf_size // 2, surf_size // 2

        # -- DESCENT STAGE (Gold Octagon-ish) --
        # Main body
        pygame.draw.rect(lander_surf, GOLD, (cx - w / 2, cy, w, h / 1.5))
        # Legs (4 legs, simplified to 2 striding)
        pygame.draw.line(lander_surf, GOLD, (cx - w / 2, cy + h / 2), (cx - w / 2 - 10, cy + h), 3)  # Left Leg
        pygame.draw.line(lander_surf, GOLD, (cx + w / 2, cy + h / 2), (cx + w / 2 + 10, cy + h), 3)  # Right Leg
        # Footpads
        pygame.draw.circle(lander_surf, GREY, (int(cx - w / 2 - 10), int(cy + h)), 4)
        pygame.draw.circle(lander_surf, GREY, (int(cx + w / 2 + 10), int(cy + h)), 4)

        # -- ASCENT STAGE (Grey/Black Top) --
        pygame.draw.rect(lander_surf, DARK_GREY, (cx - w / 2.2, cy - h / 2, w / 1.1, h / 2))
        pygame.draw.polygon(lander_surf, BLACK, [(cx - w / 2.2, cy - h / 2), (cx, cy - h), (cx + w / 2.2, cy - h / 2)])

        # -- FLAMES --
        if self.env.fuel > 0:
            if action == 2:  # Main Engine
                pygame.draw.polygon(lander_surf, ORANGE,
                                    [(cx - w / 3, cy + h / 1.5), (cx + w / 3, cy + h / 1.5), (cx, cy + h * 1.5)])
            if action == 1:  # Right RCS (Visualized on side)
                pygame.draw.circle(lander_surf, WHITE, (int(cx - w / 2), int(cy - h / 4)), 3)
            if action == 3:  # Left RCS
                pygame.draw.circle(lander_surf, WHITE, (int(cx + w / 2), int(cy - h / 4)), 3)

        # Rotate and Blit
        rot_surf = pygame.transform.rotate(lander_surf, math.degrees(self.env.theta))
        rect = rot_surf.get_rect()
        rect.center = self.world_to_screen(self.env.x, self.env.y)
        self.screen.blit(rot_surf, rect)

        # 6. HUD / Telemetry
        telemetry = [
            f"ALTITUDE: {self.env.y:.1f} m",
            f"H-SPEED:  {self.env.vx:.1f} m/s",
            f"V-SPEED:  {self.env.vy:.1f} m/s",
            f"ANGLE:    {math.degrees(self.env.theta):.1f} deg",
            f"FUEL:     {self.env.fuel:.1f} kg",
            f"MASS:     {self.env.mass:.0f} kg"
        ]

        for i, line in enumerate(telemetry):
            color = WHITE
            if "FUEL" in line and self.env.fuel < 50: color = RED
            txt = self.font.render(line, True, color)
            self.screen.blit(txt, (10, 10 + i * 20))

        if self.env.landed:
            s = pygame.Surface((SCREEN_WIDTH, 60), pygame.SRCALPHA)
            s.fill((0, 255, 0, 100))
            self.screen.blit(s, (0, SCREEN_HEIGHT / 2 - 30))
            msg = self.font.render("TOUCHDOWN CONFIRMED - R to Reset", True, WHITE)
            self.screen.blit(msg, (SCREEN_WIDTH / 2 - 150, SCREEN_HEIGHT / 2 - 10))

        if self.env.crashed:
            s = pygame.Surface((SCREEN_WIDTH, 60), pygame.SRCALPHA)
            s.fill((255, 0, 0, 100))
            self.screen.blit(s, (0, SCREEN_HEIGHT / 2 - 30))
            crash_msg = f"VEHICLE CRASHED: {self.env.crash_reason} - R to Reset" if self.env.crash_reason else "VEHICLE LOST - R to Reset"
            msg = self.font.render(crash_msg, True, WHITE)
            self.screen.blit(msg, (SCREEN_WIDTH / 2 - 200, SCREEN_HEIGHT / 2 - 10))

        pygame.display.flip()


def main():
    # env = LunarLanderEnv()
    env = SimpleLunarLanderEnv()
    renderer = Renderer(env)
    running = True

    while running:
        renderer.clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r: env.reset()

        keys = pygame.key.get_pressed()
        action = 0
        if keys[pygame.K_UP]:
            action = 1
        elif keys[pygame.K_LEFT]:
            action = 2
        elif keys[pygame.K_RIGHT]:
            action = 3

        env.step(action)
        renderer.render(action)

    pygame.quit()


if __name__ == "__main__":
    main()