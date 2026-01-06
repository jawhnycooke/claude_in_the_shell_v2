"""Agent configuration options."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AgentConfig:
    """
    Configuration for the agent loop.

    Attributes:
        model: Claude model to use
        name: Agent name
        max_tokens: Maximum tokens per response
        temperature: Sampling temperature
        enable_voice: Enable voice pipeline
        enable_motion: Enable motion control
        mock_hardware: Use mock hardware instead of SDK
        system_prompt_path: Path to system prompt file
        persona_path: Optional path to persona prompt file
    """

    model: str = "claude-haiku-4-5-20251001"
    name: str = "Jarvis"
    max_tokens: int = 4096
    temperature: float = 0.7

    # Component flags
    enable_voice: bool = False
    enable_motion: bool = True
    mock_hardware: bool = False

    # Paths
    system_prompt_path: str = "prompts/system.md"
    persona_path: str | None = None

    def load_system_prompt(self) -> str:
        """
        Load system prompt from file, optionally overlaying persona.

        Returns:
            Complete system prompt text

        Raises:
            FileNotFoundError: If prompt file doesn't exist
        """
        base_prompt = Path(self.system_prompt_path).read_text()

        if self.persona_path:
            persona_prompt = Path(self.persona_path).read_text()
            # Extract content after YAML frontmatter
            if "---" in persona_prompt:
                _, _, persona_content = persona_prompt.split("---", 2)
                base_prompt = f"{base_prompt}\n\n{persona_content.strip()}"

        return base_prompt
