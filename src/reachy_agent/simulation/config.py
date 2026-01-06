"""Configuration models for MuJoCo simulation.

This module provides Pydantic models for simulation configuration.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field


class RenderQuality(str, Enum):
    """Render quality presets."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RenderConfig(BaseModel):
    """Configuration for rendering.

    Attributes:
        width: Render width in pixels
        height: Render height in pixels
        quality: Quality preset
        shadows: Enable shadow rendering
        antialiasing: Enable antialiasing
    """

    width: int = Field(default=640, ge=64, le=4096)
    height: int = Field(default=480, ge=64, le=4096)
    quality: RenderQuality = RenderQuality.MEDIUM
    shadows: bool = True
    antialiasing: bool = True


class PhysicsConfig(BaseModel):
    """Configuration for physics simulation.

    Attributes:
        gravity: Gravity vector (x, y, z) in m/s²
        friction: Default friction coefficient
        damping: Default joint damping
    """

    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    friction: float = Field(default=0.8, ge=0.0, le=2.0)
    damping: float = Field(default=1.0, ge=0.0, le=100.0)


class RecordingConfig(BaseModel):
    """Configuration for video recording.

    Attributes:
        fps: Frames per second
        format: Video format (mp4, gif, etc.)
    """

    fps: int = Field(default=30, ge=1, le=120)
    format: str = Field(default="mp4")


class SimulationConfig(BaseModel):
    """Complete simulation configuration.

    Attributes:
        model_path: Path to MJCF model file
        timestep: Physics timestep in seconds (default 0.002 for 500Hz)
        n_substeps: Number of physics substeps per step
        realtime: Run simulation in real-time
        viewer: Enable visualization window
        headless: Use headless rendering (no display)
        render: Render configuration
        physics: Physics configuration
        recording: Recording configuration

    Example:
        >>> config = SimulationConfig(
        ...     timestep=0.002,
        ...     realtime=True,
        ...     viewer=True,
        ... )
    """

    model_path: str = Field(
        default="data/models/reachy_mini/reachy_mini.xml",
        description="Path to MJCF model file",
    )
    timestep: Annotated[float, Field(ge=0.0001, le=0.1)] = 0.002
    n_substeps: Annotated[int, Field(ge=1, le=100)] = 4
    realtime: bool = True
    viewer: bool = False
    headless: bool = False
    render: RenderConfig = Field(default_factory=RenderConfig)
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    recording: RecordingConfig = Field(default_factory=RecordingConfig)

    @property
    def effective_timestep(self) -> float:
        """Get effective timestep including substeps."""
        return self.timestep / self.n_substeps

    @property
    def physics_hz(self) -> float:
        """Get physics frequency in Hz."""
        return 1.0 / self.effective_timestep

    def get_model_path(self) -> Path:
        """Get model path as Path object."""
        return Path(self.model_path)

    @classmethod
    def from_yaml(cls, config: dict) -> "SimulationConfig":
        """Create config from YAML dict.

        Args:
            config: Configuration dict from YAML

        Returns:
            SimulationConfig instance
        """
        sim_config = config.get("simulation", {})
        return cls(**sim_config)


class DomainRandomizationConfig(BaseModel):
    """Configuration for domain randomization.

    Used for sim-to-real transfer and robust policy training.

    Attributes:
        enabled: Enable domain randomization
        mass_range: Range for mass scaling (min, max)
        friction_range: Range for friction coefficient
        damping_range: Range for damping coefficient
        noise_std: Standard deviation for sensor noise
    """

    enabled: bool = False
    mass_range: tuple[float, float] = (0.8, 1.2)
    friction_range: tuple[float, float] = (0.5, 1.2)
    damping_range: tuple[float, float] = (0.5, 2.0)
    noise_std: float = 0.01


class ScenarioConfig(BaseModel):
    """Configuration for simulation scenarios.

    Defines the initial state and environment for a simulation.

    Attributes:
        name: Scenario name
        initial_pose: Initial joint positions (degrees)
        objects: List of objects to spawn in environment
        lighting: Lighting configuration
    """

    name: str = "default"
    initial_pose: dict[str, float] = Field(default_factory=dict)
    objects: list[dict] = Field(default_factory=list)
    lighting: dict = Field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "ScenarioConfig":
        """Load scenario from YAML file.

        Args:
            path: Path to scenario YAML file

        Returns:
            ScenarioConfig instance
        """
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
