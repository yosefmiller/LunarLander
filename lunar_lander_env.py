from dataclasses import dataclass
from typing import Tuple, List, Type, Iterable, TypedDict
from enum import Flag

import numpy as np
import pygame
import math

from PIL import Image

# --- Constants ---
FPS = 60
DT = 1.0 / FPS  # Seconds

# PHYSICS CONSTANTS (MOON)
GRAVITY = -1.625  # Moon gravity (approx 1/6 Earth)
SCALE = 10.0  # Pixels per meter (Zoomed out to show approach)

# WORLD
SCREEN_WIDTH = 1000  # Pixels
SCREEN_HEIGHT = 600  # Pixels
GROUND_Y = 50.0  # Pixels from bottom of screen (visual offset)

# LANDER PROPERTIES
LANDER_DRY_MASS = 600.0  # kg (Structure + Descent Engine)
FUEL_MASS_START = 100.0  # kg (Propellant)

LANDER_WIDTH = 6.0  # Meters
LANDER_HEIGHT = 4.0  # Meters

# THRUST PARAMETERS
# Weight (Full) = 1000kg * 1.625 = 1625 N
# Max Thrust should be ~2-3x Weight for control.
# Apollo Descent Engine was throttlable, here we simulate 'max' burst.
MAIN_THRUST = 4500.0  # T/W Ratio ~ 2.7 (at start) -> ~ 4.6 (when empty)
SIDE_THRUST = 1000.0  # RCS Thrusters
SIDE_ENGINE_OFFSET = 3.0  # Torque leverage

FUEL_CONSUMPTION_MAIN = 2.0  # kg per second
FUEL_CONSUMPTION_SIDE = 0.5  # kg per second

# LANDING ZONE
PAD_WIDTH = 30.0  # Meters
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

class CrashReason(Flag):
    TILTED = 1
    MISSED_LZ = 2
    TOO_FAST = 4
    OUT_OF_RANGE = 8
    # TIMEOUT = 16

    def __str__(self):
        reasons = []
        if CrashReason.TILTED in self: reasons.append("Tilted")
        if CrashReason.MISSED_LZ in self: reasons.append("Missed LZ")
        if CrashReason.TOO_FAST in self: reasons.append("Too Fast")
        if CrashReason.OUT_OF_RANGE in self: reasons.append("Out of Range")
        # if CrashReason.TIMEOUT in self: reasons.append("Timeout")
        return ", ".join(reasons) if reasons else ""

@dataclass
class LunarLanderState:
    """Represents the state of the lunar lander."""
    x: float | np.float64  # Relative distance X (normalized)
    y: float | np.float64  # Relative distance Y (normalized)
    vx: float | np.float64  # Velocity X (normalized)
    vy: float | np.float64  # Velocity Y (normalized)
    theta: float | np.float64  # Angle (radians)
    omega: float | np.float64  # Angular velocity
    fuel: float  # Fuel remaining (normalized)
    on_pad: float  # Pad contact sensor (0.0 or 1.0)

    def to_array(self) -> np.ndarray:
        """Convert state to numpy array for RL algorithms."""
        return np.array([
            self.x, self.y, self.vx, self.vy,
            self.theta, self.omega, self.fuel, # self.on_pad
        ], dtype=np.float32)

    def __str__(self):
        return f"LunarLanderState(x={self.x:.1f}, y={self.y:.1f}, vx={self.vx:.1f}, vy={self.vy:.1f}, theta={math.degrees(self.theta):.1f} deg, fuel={self.fuel:.1f} kg, on_pad={self.on_pad})"

class EpisodeResult(TypedDict):
    reward: float|None  # Filled after rewards are accumulated at the end of the episode
    steps: int
    landed: bool
    timeout: bool
    crashed: bool
    crash_reason: CrashReason

class LunarLanderEnv:
    x: float | np.float64
    y: float | np.float64
    vx: float | np.float64
    vy: float | np.float64
    theta: float | np.float64
    omega: float | np.float64
    fuel: float
    mass: float
    landed: bool
    crashed: bool
    crash_reason: CrashReason
    number_of_steps: int
    timeout: bool
    trace: list
    prev_action: int
    prev_shaping: float|None
    moment_of_inertia: float


    def __init__(self, max_number_of_seconds=25, debug=False):
        self.max_number_of_seconds = float(max_number_of_seconds)
        self.debug=debug
        self.reset()
        if debug:
            print(f"Initial Reward Potential: {self._calculate_shaping():.2f}")

    def reset(self) -> LunarLanderState:
        self._init_state()

        self.fuel = FUEL_MASS_START
        self.mass = LANDER_DRY_MASS + self.fuel
        self.prev_action = 0

        self.number_of_steps = 0
        self.landed = False
        self.crashed = False
        self.crash_reason = CrashReason(0)  # Store reason for display
        self.timeout = False
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
        """
        Calculates a potential-based shaping value.
        Higher (closer to 0) is better.
        The shaping stays bounded and well below the landing bonus.
        """
        # Scale state to roughly 0.0 to 1.0 range
        dist     = np.sqrt(self.x ** 2 + (self.y-3) ** 2) / 75.0  # 1 unit = 75m
        velocity = np.sqrt(self.vx ** 2 + self.vy ** 2) / 15.0    # 1 unit = 15m/s
        tilt     = abs(self.theta) * 2 / math.pi                  # 1 unit = 90degrees
        omega    = abs(self.omega) * 2 / math.pi                  # 1 unit = 90degrees/s

        # y = np.clip((self.y-3) / 50.0, 0.0, 1.0)  # Altitude factor (0 at ground, 1 at start)
        # proximity_factor = (1.0 - y) ** 2  # Ramps up as we get closer to the ground

        # Calculate potential
        # The sum of these weights is the maximum potential you can gain over an episode.
        # This sum must be less than the landing bonus to ensure the agent will not intentionally crash for shaping rewards.
        return (- 20.0 * dist
                - 10.0 * velocity
                - 5.0 * (tilt > 90) * (tilt - 90)
        )

    def _calculate_reward(self, action: int) -> float:
        """
        Calculates the reward for the current transition.
        Assumes the physics step has completed and termination flags
        (landed, crashed, timeout) are already updated.
        """

        # 1. Time Penalty
        reward = -0.0/FPS

        # 2. Potential-Based Shaping (Dense)
        shaping = self._calculate_shaping()
        if self.prev_shaping is not None:
            # Reward is the difference in potential
            reward += shaping - self.prev_shaping
        self.prev_shaping = shaping

        # 3. Control/Fuel Penalties (Mild)
        # if action == 1:
        #     reward -= FUEL_CONSUMPTION_MAIN * DT * 0.5
        # elif action in [2, 3]:
        #     reward -= FUEL_CONSUMPTION_SIDE * DT * 0.5

        # 4. Terminal Conditions (Sparse)
        if self.landed:
            reward += 500.0

            # Add bonus for landing squarely on both feet
            if abs(self.theta) < 0.1:
                reward += 20.0

            # Add fuel penalty
            reward -= (FUEL_MASS_START - self.fuel)*2

        elif self.crashed:
            # reward -= 100.0
            if CrashReason.TOO_FAST in self.crash_reason:
                reward -= 20.0
            if CrashReason.MISSED_LZ in self.crash_reason:
                reward -= 20.0
            if CrashReason.TILTED in self.crash_reason:
                reward -= 10.0
            if CrashReason.OUT_OF_RANGE in self.crash_reason:
                reward -= 50.0

        elif self.timeout:
            reward -= 5.0

        return reward

    def step(self, action: int) -> Tuple[LunarLanderState, float, bool, EpisodeResult]:
        """
        Action space:
        0: Do nothing
        1: Main Engine
        2: Left Engine (Rotates CW)
        3: Right Engine (Rotates CCW)

        :returns
        state: LunarLanderState
        reward: float
        done: bool
        result: dict (crash status, landing status, max step exceeded, crash reason)
        """
        if self.landed or self.crashed or self.timeout:
            return self._get_state(), 0, True, EpisodeResult(steps=self.number_of_steps, reward=0, landed=self.landed, crashed=self.crashed, timeout=self.timeout, crash_reason=self.crash_reason)

        #########################
        ##### APPLY THRUST ######
        #########################
        self.mass = LANDER_DRY_MASS + self.fuel

        force_x = 0.0
        force_y = self.mass * GRAVITY  # Weight
        torque = 0.0

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

        ############################
        ##### TERMINAL CHECKS #####
        ############################
        done = False

        # Legs: Define the LEM feet relative to center
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
            angle_safe = abs(self.theta) < 0.5
            on_pad = abs(self.x) < (PAD_WIDTH / 2.0)

            # Physics adjustment to sit on ground
            self.y -= min(left_foot[1], right_foot[1])
            if vel_safe and angle_safe and on_pad:
                self.landed = True
                if self.debug:
                    print(f"EAGLE HAS LANDED. Fuel Spent: {FUEL_MASS_START - self.fuel:.1f} kg")
            else:
                self.crashed = True
                if not vel_safe: self.crash_reason |= CrashReason.TOO_FAST
                if not angle_safe: self.crash_reason |= CrashReason.TILTED
                if not on_pad: self.crash_reason |= CrashReason.MISSED_LZ
                # print(f"ABORT/CRASH: {self.crash_reason}")

        # Unsafe descent conditions
        # if abs(self.theta) > (math.pi / 2.0):
        #     done = True
        #     self.crashed = True
        #     self.crash_reason |= CrashReason.TILTED

        # Out of bounds
        if abs(self.x) > (SCREEN_WIDTH / SCALE) / 2 + 20 or self.y > (SCREEN_HEIGHT / SCALE) + 20:
            done = True
            self.crashed = True
            self.crash_reason = CrashReason.OUT_OF_RANGE

        # Limit the number of steps to prevent infinite episodes
        self.number_of_steps += 1
        if self.number_of_steps*DT >= self.max_number_of_seconds and not done:
            done = True
            self.timeout = True

        ############################
        ##### CALCULATE REWARD #####
        ############################
        reward = self._calculate_reward(action)

        result = EpisodeResult(
            reward=None,
            crashed=self.crashed,
            landed=self.landed,
            timeout=self.timeout,
            crash_reason=self.crash_reason,
            steps=self.number_of_steps,
        )
        return self._get_state(), reward, done, result

    def _get_state(self) -> LunarLanderState:
        # Normalize state variables to roughly -1.0 to 1.0 range for RL algorithms.
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
    def __init__(self, num_actions=4, **kwargs):
        self.num_actions = num_actions  # Default is only Main Engine and No Action
        super().__init__(**kwargs)

    def _init_state(self):
        """Override _init_sate() to provide simpler initial conditions."""
        # Simple initialization: straight drop
        self.x = 0.0  # Start centered on the pad
        
        self.y = np.random.uniform(40, 50)  # Start high up

        # No lateral velocity
        self.vx = 0.0

        # Randomized downward velocity (steeper than complex env)
        self.vy = np.random.uniform(-3.0, -0.5)

        # No initial tilt
        self.theta = 0.0
        self.omega = 0.0
    
    def step(self, action):
        if self.num_actions == 2 and action not in [0, 1]:
            raise ValueError("Invalid action for SimpleLunarLanderEnv. Only 0 (No Action) and 1 (Main Engine) are allowed.")
        return super().step(action)

class LLE_XOffset(LunarLanderEnv):
    """
    Simplified lunar lander environment: drops down from some offset for x, y (no tilt).
    Inherits from LunarLanderEnv to avoid code duplication.

    Key differences:
    - Randomized x offset (vy: -10.0 to 10.0 m/s)
    - No lateral velocity (vx=0) instead of 5-10 m/s
    - No initial tilt (theta=0) instead of random ±0.2 rad
    - Randomized downward velocity (vy: -3.0 to -0.5 m/s)
    """
    def _init_state(self):
        """Override _init_sate() to provide simpler initial conditions."""
        # Add an x offset
        start_x_range = [-15.0, -10.0] if np.random.rand() > 0.5 else [10.0, 15.0]
        self.x = np.random.uniform(start_x_range[0], start_x_range[1])
        
        self.y = np.random.uniform(40, 50)  # Start high up

        # No lateral velocity
        self.vx = 0.0

        # Randomized downward velocity (steeper than complex env)
        self.vy = np.random.uniform(-1.5, -0.5)

        # No initial tilt
        self.theta = 0.0
        self.omega = 0.0

class LLE_InitialVelocity(LunarLanderEnv):
    """
    Harder lunar lander environment: drops down from some offset for x, y and some initial velociy (no tilt).
    Inherits from LunarLanderEnv to avoid code duplication.

    Key differences:
    - Randomized x offset (vy: -1.5 to 1.5.0 m/s)
    - Randomized velocity (vx: instead of -3.0 to 3.0 m/s)
    - No initial tilt (theta=0)
    - Randomized downward velocity (vy: -3.0 to -0.5 m/s)
    """

    def _init_state(self):
        """Override _init_sate() to provide simpler initial conditions."""
        # Add an x offset
        start_x_range = [-30, -20] if np.random.rand() > 0.5 else [20, 30]
        self.x = np.random.uniform(start_x_range[0], start_x_range[1])
        
        self.y = np.random.uniform(40, 50)  # Start high up

        # Horizontal velocity points towards the center, but fast
        direction = -1.0 if self.x > 0 else 1.0
        self.vx = np.random.uniform(1.0, 3.0) * direction

        # Randomized downward velocity (steeper than complex env)
        self.vy = np.random.uniform(-1.5, -0.5)

        # No initial tilt
        self.theta = 0.0
        self.omega = 0.0

class RandomLunarLander(LunarLanderEnv):
    """
    Fully randomized lunar lander environment: some offset x, y, tilt, dx, dy
    """

    def _init_state(self):
        """Override _init_sate() to provide simpler initial conditions."""
        self.x = np.random.uniform(-50, 50)
        self.y = np.random.uniform(20, 50)
        self.vx = np.random.uniform(-5.0, -1.0) * self.x / abs(self.x)  # Velocity points towards center
        self.vy = np.random.uniform(-2.5, -0.5)
        self.theta = np.random.uniform(-0.2, 0.2)
        self.omega = 0.0

class VectorizedEnv:
    """
    Runs multiple environments in parallel.
    """
    envs: List[LunarLanderEnv]

    def __init__(self, env_fn: Type[LunarLanderEnv], num_envs=8, debug=False):
        self.num_envs = num_envs
        self.envs = [env_fn(debug=debug) for _ in range(num_envs)]

    def reset(self) -> np.ndarray:
        return np.array([env.reset().to_array() for env in self.envs])

    def reset_env(self, env_index) -> np.ndarray:
        return self.envs[env_index].reset().to_array()

    def step(self, actions: Iterable[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[EpisodeResult]]:
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        obs     = np.array([r[0].to_array() for r in results])
        rewards = np.array([r[1] for r in results])
        dones   = np.array([r[2] for r in results])
        infos   = [r[3] for r in results]

        # Auto-reset done environments
        for i, done in enumerate(dones):
            if done:
                obs[i] = self.reset_env(i)

        return obs, rewards, dones, infos

class Renderer:
    def __init__(self, env, save_gif=False):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Apollo Lunar Descent Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 16)
        self.env = env
        self.should_save_gif = save_gif
        self.frames = []

    @staticmethod
    def world_to_screen(x, y):
        # Center x on screen (Screen Center = World 0)
        screen_x = int(SCREEN_WIDTH / 2 + x * SCALE)
        # Flip Y (Screen 0 is top)
        screen_y = int(SCREEN_HEIGHT - (GROUND_Y + y * SCALE))
        return screen_x, screen_y

    def render(self, action=0, bins=None):
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

        # Draw grids if bins provided
        if bins is not None:
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
            if action == 1:  # Main Engine
                pygame.draw.polygon(lander_surf, ORANGE,
                                    [(cx - w / 3, cy + h / 1.5), (cx + w / 3, cy + h / 1.5), (cx, cy + h * 1.5)])
            if action == 2:  # Right RCS
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
            f"ANG.VEL:  {math.degrees(self.env.omega):.1f} deg/s",
            f"FUEL:     {self.env.fuel:.1f} kg",
            f"MASS:     {self.env.mass:.0f} kg",
            f"SHAPING:  {self.env._calculate_shaping():.2f}",
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

        if self.env.crashed or self.env.timeout:
            s = pygame.Surface((SCREEN_WIDTH, 60), pygame.SRCALPHA)
            s.fill((255, 0, 0, 100))
            self.screen.blit(s, (0, SCREEN_HEIGHT / 2 - 30))
            crash_msg = f"VEHICLE CRASHED: {self.env.crash_reason} - R to Reset" if self.env.crashed else "VEHICLE TIMEOUT - R to Reset"
            msg = self.font.render(crash_msg, True, WHITE)
            self.screen.blit(msg, (SCREEN_WIDTH / 2 - 200, SCREEN_HEIGHT / 2 - 10))

        pygame.display.flip()

        # Capture frame for GIF if enabled
        if self.should_save_gif:
            # Convert pygame surface to PIL Image in a robust way.
            # pygame.image.tostring is the commonly used API to get a bytes object
            # in a known pixel ordering; Image.frombuffer avoids an extra copy.
            # Use the newer API name `tobytes` (preferred in recent pygame versions)
            string_image = pygame.image.tobytes(self.screen, 'RGB')
            pil_image = Image.frombuffer('RGB', (SCREEN_WIDTH, SCREEN_HEIGHT), string_image, 'raw', 'RGB', 0, 1)
            # Use a copy to decouple from the underlying buffer and keep an RGBA image
            self.frames.append(pil_image.convert('RGBA').copy())

    def save_gif(self, gif_path: str):
        """Save captured frames to a GIF.

        Notes:
        - Many GIF viewers impose a minimum frame delay (commonly ~20 ms). Very small
          durations (e.g. 16 ms for 60 FPS) can be rounded or clamped which makes the
          animation appear slower than expected. To improve compatibility we enforce
          a small minimum delay (20 ms) and convert frames to an adaptive palette.
        - Converting frames to 'P' (paletted) with an adaptive palette reduces color
          artifacts and keeps file size reasonable.
        """
        if self.should_save_gif and self.frames:
            print(f"Saving GIF with {len(self.frames)} frames to {gif_path}...")

            # Convert to paletted images using an adaptive palette (better compatibility)
            paletted = [f.convert('P', palette=Image.Palette.ADAPTIVE, colors=256) for f in self.frames]

            # Compute duration in milliseconds. Use a small minimum (20ms) because
            # many viewers clamp very short delays, which can make the GIF look slow.
            duration_ms = max(20, int(round(1000.0 / FPS)))

            # Save with disposal=2 (restore to background) to avoid frame accumulation
            # and optimize palette. Provide a consistent per-frame duration.
            try:
                paletted[0].save(
                    gif_path,
                    save_all=True,
                    append_images=paletted[1:],
                    duration=duration_ms,
                    loop=0,
                    disposal=2,
                    optimize=True,
                )
            except TypeError:
                # Older Pillow versions may not accept disposal/optimize kwargs; fall back
                paletted[0].save(gif_path, save_all=True, append_images=paletted[1:], duration=duration_ms, loop=0)

            # Clear frames after saving
            self.frames = []

def main():
    env = LunarLanderEnv()
    # env = SimpleLunarLanderEnv()
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

        if keys[pygame.K_1]:
            env = SimpleLunarLanderEnv()
            renderer = Renderer(env)
            continue
        elif keys[pygame.K_2]:
            env = LLE_XOffset()
            renderer = Renderer(env)
            continue
        elif keys[pygame.K_3]:
            env = LLE_InitialVelocity()
            renderer = Renderer(env)
            continue
        elif keys[pygame.K_4]:
            env = LunarLanderEnv()
            renderer = Renderer(env)
            continue
        elif keys[pygame.K_5]:
            env = RandomLunarLander()
            renderer = Renderer(env)
            continue

        if keys[pygame.K_q]:
            break

        env.step(action)
        renderer.render(action)

    pygame.quit()


if __name__ == "__main__":
    main()