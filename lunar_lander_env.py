from dataclasses import dataclass
from typing import Tuple, List, Type, Iterable, TypedDict
from enum import Flag

import numpy as np
import math

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


    def __init__(self, max_number_of_steps=25*60, debug=False, pad_x_offset=0.0):
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

    def calculate_shaping(self) -> float:
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
        # tilt     = abs(self.theta) * 2 / math.pi                  # 1 unit = 90degrees
        # omega    = abs(self.omega) * 2 / math.pi                  # 1 unit = 90degrees/s

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

    def calculate_reward(self, action: int) -> float:
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
        shaping = self.calculate_shaping()
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
        reward = self.calculate_reward(action)

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
