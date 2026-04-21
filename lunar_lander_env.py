from dataclasses import dataclass
from typing import Tuple, List, Type, Iterable, TypedDict
from enum import Flag

import numpy as np
import pygame
import math
from datetime import datetime
import imageio
import os

# Additional graphics
NASA_LOGO = pygame.image.load('graphics/nasa_logo.png')
NASA_LOGO = pygame.transform.scale(NASA_LOGO, (30, 25))  # Resize
ASTRONAUT = pygame.image.load('graphics/astronaut.png')
ASTRONAUT = pygame.transform.scale(ASTRONAUT, (45, 50))  # Resize
AMERICAN_FLAG = pygame.image.load('graphics/american_flag.png')
AMERICAN_FLAG = pygame.transform.scale(AMERICAN_FLAG, (22, 15))  # Resize

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
PAD_WIDTH = 20.0  # Meters
PAD_X_TARGET = 0.0  # Center of the world (0,0)

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

# Action Space
ACTIONS = {0: 'do nothing', 1: 'use main engine', 2: 'use left engine', 3: 'use right engine'}

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
            self.theta, self.omega, self.fuel, self.on_pad
        ], dtype=np.float32)

    def __str__(self):
        return f"LunarLanderState(x={self.x:.1f}, y={self.y:.1f}, vx={self.vx:.1f}, vy={self.vy:.1f}, theta={math.degrees(self.theta):.1f} deg, fuel={self.fuel:.1f} kg, on_pad={self.on_pad})"

class EpisodeResult(TypedDict):
    reward: float|None  # Filled after rewards are accumulated at the end of the episode
    steps: int
    spent: float
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
    record_trace: bool
    prev_action: int
    prev_shaping: float|None
    moment_of_inertia: float


    def __init__(self, max_number_of_steps=1000, debug=False, pad_x_offset=0.0):
        """
        Initialize Lunar Lander Environment

        Args:
            max_number_of_steps: Maximum steps before episode terminates
            debug: Enable debug printing
            pad_x_offset: X-offset of landing pad from center in meters
        """
        self.action_space = ACTIONS
        self.max_number_of_steps = max_number_of_steps
        self.debug=debug
        self.pad_x_offset = pad_x_offset
        self.reset()

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
        self.record_trace = False

        self.prev_shaping = None

        # Approximate Moment of Inertia (Rectangle)
        # Note: In reality, `I` changes as fuel burns, but we keep `I` constant for stability
        self.moment_of_inertia = LANDER_DRY_MASS * (LANDER_WIDTH ** 2 + LANDER_HEIGHT ** 2) / 12.0

        return self._get_state()

    def _init_state(self):
        # 1. INITIALIZATION (Apollo "High Gate" Style)
        rng = np.random.default_rng()
        start_x_range = [-60, -40] if rng.uniform() > 0.5 else [40, 60]
        self.x = rng.uniform(start_x_range[0], start_x_range[1])
        self.y = rng.uniform(40, 50)  # Start high up

        # Velocity points towards the center, but fast
        direction = -1.0 if self.x > 0 else 1.0
        self.vx = rng.uniform(5.0, 10.0) * direction
        self.vy = rng.uniform(-2.0, 0.0)  # Slight downward drift

        # Random initial tilt (imperfection)
        self.theta = rng.uniform(-0.2, 0.2)
        self.omega = 0.0

    def _calculate_shaping(self) -> float:
        """
        Goal-oriented potential: measures "distance" to the desired final state.
        Higher (closer to 0) = closer to: on pad, at rest, upright.
        """
        # Adjust target to account for pad offset
        dist_x = self.x - self.pad_x_offset
        dist_y = self.y - 3.0

        # Scale state to roughly 0.0 to 1.0 range
        dist     = np.sqrt(dist_x ** 2 + dist_y ** 2) / 75.0  # 1 unit = 75m
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
                # - 3.0 * (tilt > 1) * (tilt - 1)
                # - 3.0 * (omega > 0.5)
        )

    def _calculate_reward(self, action: int) -> float:
        """
        Calculate the reward for the current transition.

        Reward structure:
        1. Small time penalty (encourages faster landing)
        2. Energy-based shaping (dense feedback for progress)
        3. Control penalties (mild, to encourage fuel efficiency)
        4. Terminal bonuses/penalties (sparse, dominant signal)
        """
        reward = 0.0

        # 1. Time Penalty
        # reward -= 1.0/FPS

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
            if abs(self.theta) < 0.1:  # < ~5.7 degrees
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
            return self._get_state(), 0, True, EpisodeResult(
                steps=self.number_of_steps,
                reward=0,
                spent=FUEL_MASS_START - self.fuel,
                landed=self.landed,
                crashed=self.crashed,
                timeout=self.timeout,
                crash_reason=self.crash_reason
            )

        #########################
        ##### APPLY THRUST ######
        #########################
        self.mass = LANDER_DRY_MASS + self.fuel

        force_x = 0.0
        force_y = self.mass * GRAVITY  # Weight
        torque = 0.0

        sin_theta = math.sin(self.theta)
        cos_theta = math.cos(self.theta)

        match action:
            case 0:  # No Action
                pass
            case 1:  # Main Engine
                f_thrust = MAIN_THRUST
                force_x += -sin_theta * f_thrust
                force_y += cos_theta * f_thrust
                self.fuel -= FUEL_CONSUMPTION_MAIN * DT
            case 2:  # Left RCS
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

        if self.record_trace:
            self.trace.append((self.x, self.y))
            if len(self.trace) > 200: self.trace.pop(0)

        ############################
        ##### TERMINAL CHECKS #####
        ############################
        done = False

        half_w = LANDER_WIDTH / 2.0 + 1.0
        half_h = LANDER_HEIGHT / 2.0 + 1.0

        # Only check ground contact when we're close (huge speedup)
        if self.y > 5.0:
            # Too high to touch ground, skip expensive foot calculations
            pass
        else:
            # Compute foot positions (reuse sin/cos from thrust calculation)
            # Left foot: relative position (-half_w, -half_h)
            # left_foot_x = self.x + (-half_w) * cos_theta - (-half_h) * sin_theta
            left_foot_y = self.y + (-half_w) * sin_theta + (-half_h) * cos_theta

            # Right foot: relative position (half_w, -half_h)
            # right_foot_x = self.x + half_w * cos_theta - (-half_h) * sin_theta
            right_foot_y = self.y + half_w * sin_theta + (-half_h) * cos_theta

            # Ground Plane (y=0)
            if left_foot_y <= 0 or right_foot_y <= 0:
                done = True

                # Landing Criteria
                vel_safe = abs(self.vy) < 2.5 and abs(self.vx) < 2.0
                angle_safe = abs(self.theta) < 0.3
                on_pad = abs(self.x - self.pad_x_offset) < (PAD_WIDTH / 2.0)

                # Physics adjustment to sit on ground
                self.y -= min(left_foot_y, right_foot_y)
                if vel_safe and angle_safe and on_pad:
                    self.landed = True
                    if self.debug:
                        print(f"EAGLE HAS LANDED. Fuel Spent: {FUEL_MASS_START - self.fuel:.1f} kg")
                else:
                    self.crashed = True
                    if not vel_safe: self.crash_reason |= CrashReason.TOO_FAST
                    if not angle_safe: self.crash_reason |= CrashReason.TILTED
                    if not on_pad: self.crash_reason |= CrashReason.MISSED_LZ

        # Out of bounds
        if abs(self.x) > (SCREEN_WIDTH / SCALE) / 2 + 20 or self.y > (SCREEN_HEIGHT / SCALE) + 20:
            done = True
            self.crashed = True
            self.crash_reason = CrashReason.OUT_OF_RANGE

        # Limit the number of steps to prevent infinite episodes
        self.number_of_steps += 1
        if self.number_of_steps >= self.max_number_of_steps and not done:
            done = True
            self.timeout = True

        ############################
        ##### CALCULATE REWARD #####
        ############################
        reward = self._calculate_reward(action)

        result = EpisodeResult(
            reward=None,
            spent=FUEL_MASS_START - self.fuel,
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
            x=(self.x - self.pad_x_offset) / 50.0,  # Relative to pad
            y=self.y / 50.0,  # Relative Dist Y
            vx=self.vx / 10.0,  # Vel X
            vy=self.vy / 10.0,  # Vel Y
            theta=self.theta,  # Angle
            omega=self.omega,  # Angular Vel
            fuel=self.fuel / FUEL_MASS_START,
            on_pad=1.0 if (abs(self.x - self.pad_x_offset) < PAD_WIDTH / 2) else 0.0
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
        """Override _init_state() to provide simpler initial conditions."""
        rng = np.random.default_rng()
        self.x = self.pad_x_offset  # Start centered on the pad
        self.y = rng.uniform(40, 50)
        self.vx = 0.0
        self.vy = rng.uniform(-3.0, -0.5)
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
        rng = np.random.default_rng()
        start_x_range = [-15.0, -10.0] if rng.uniform() > 0.5 else [10.0, 15.0]
        self.x = rng.uniform(start_x_range[0], start_x_range[1]) + self.pad_x_offset
        self.y = rng.uniform(40, 50)
        self.vx = 0.0
        self.vy = rng.uniform(-1.5, -0.5)
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
        rng = np.random.default_rng()
        start_x_range = [-30, -20] if rng.uniform() > 0.5 else [20, 30]
        self.x = rng.uniform(start_x_range[0], start_x_range[1]) + self.pad_x_offset
        self.y = rng.uniform(40, 50)
        
        direction = -1.0 if self.x > self.pad_x_offset else 1.0
        self.vx = rng.uniform(1.0, 3.0) * direction
        self.vy = rng.uniform(-1.5, -0.5)
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
            f"SHAPING:  {self.env._calculate_shaping():.2f}",
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