"""Configuration loading and validation using Pydantic models."""

import os
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

# --- Pydantic Configuration Models ---


class AgentConfig(BaseModel):
    """Agent configuration."""

    model: str = "claude-haiku-4-5-20251001"
    name: str = "Jarvis"
    max_tokens: int = 4096
    temperature: float = 0.7


class VoiceConfig(BaseModel):
    """Voice pipeline configuration."""

    personas: list[str] = Field(default_factory=lambda: ["motoko", "batou", "jarvis"])
    wake_sensitivity: float = 0.5
    silence_threshold: float = 0.3
    max_listen_time: float = 30.0


class IdleConfig(BaseModel):
    """Idle behavior configuration."""

    speed: float = 0.1
    amplitude: float = 0.3
    antenna_drift: float = 0.2


class WobbleConfig(BaseModel):
    """Speech wobble configuration."""

    intensity: float = 1.0
    frequency: float = 4.0


class MotionConfig(BaseModel):
    """Motion control configuration."""

    tick_hz: int = 30
    idle: IdleConfig = Field(default_factory=IdleConfig)
    wobble: WobbleConfig = Field(default_factory=WobbleConfig)


class MemoryConfig(BaseModel):
    """Memory system configuration."""

    path: str = "~/.reachy/memory"
    context_window_size: int = 5
    embedding_model: str = "all-MiniLM-L6-v2"
    cleanup_interval: int = 3600


class RobotConfig(BaseModel):
    """Robot backend configuration."""

    backend: str = "sdk"
    connect_timeout: float = 5.0


class CacheConfig(BaseModel):
    """Cache configuration for MCP tools."""

    status_ttl: float = 0.2
    pose_ttl: float = 0.2
    battery_ttl: float = 5.0
    sensor_ttl: float = 0.1


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""

    realtime_model: str = "gpt-4o-realtime-preview"


class ReachyConfig(BaseModel):
    """Root configuration model for the Reachy Agent."""

    agent: AgentConfig = Field(default_factory=AgentConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    motion: MotionConfig = Field(default_factory=MotionConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    robot: RobotConfig = Field(default_factory=RobotConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    system_prompt: str = "prompts/system.md"


# --- Configuration Loading Functions ---


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """
    Apply environment variable overrides to configuration.

    Environment variables follow the pattern: REACHY_SECTION_KEY
    Example: REACHY_AGENT_MODEL overrides agent.model

    Args:
        config: Configuration dictionary to modify

    Returns:
        Modified configuration dictionary
    """
    env_mapping = {
        "REACHY_AGENT_MODEL": ("agent", "model"),
        "REACHY_AGENT_NAME": ("agent", "name"),
        "REACHY_MEMORY_PATH": ("memory", "path"),
        "REACHY_ROBOT_BACKEND": ("robot", "backend"),
        "REACHY_MOTION_TICK_HZ": ("motion", "tick_hz"),
    }

    for env_var, (section, key) in env_mapping.items():
        env_value = os.environ.get(env_var)
        if env_value is not None:
            if section not in config:
                config[section] = {}

            # Handle type conversion for numeric values
            final_value: str | int | float = env_value
            if key in ("tick_hz",):
                final_value = int(env_value)
            elif key in ("temperature", "connect_timeout"):
                final_value = float(env_value)

            config[section][key] = final_value

    return config


def load_yaml(config_path: str | Path) -> dict[str, Any]:
    """
    Load raw YAML configuration from file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    return config or {}


def load_config(config_path: str = "config/default.yaml") -> ReachyConfig:
    """
    Load and validate configuration from YAML file.

    Applies environment variable overrides and validates using Pydantic.

    Args:
        config_path: Path to configuration file

    Returns:
        Validated ReachyConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
        pydantic.ValidationError: If configuration is invalid

    Examples:
        >>> config = load_config()
        >>> config.agent.model
        "claude-haiku-4-5-20251001"
        >>> config.motion.tick_hz
        30
    """
    raw_config = load_yaml(config_path)
    config_with_overrides = _apply_env_overrides(raw_config)
    return ReachyConfig(**config_with_overrides)


def load_config_dict(config_path: str = "config/default.yaml") -> dict[str, Any]:
    """
    Load configuration as a dictionary (for backwards compatibility).

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary

    Examples:
        >>> config = load_config_dict()
        >>> config["agent"]["model"]
        "claude-haiku-4-5-20251001"
    """
    raw_config = load_yaml(config_path)
    return _apply_env_overrides(raw_config)


def load_permissions(config_path: str = "config/permissions.yaml") -> dict[str, Any]:
    """
    Load permission rules from YAML file.

    Args:
        config_path: Path to permissions config

    Returns:
        Permissions configuration

    Raises:
        FileNotFoundError: If config file doesn't exist

    Examples:
        >>> perms = load_permissions()
        >>> perms["default_tier"]
        "confirm"
    """
    return load_yaml(config_path)
