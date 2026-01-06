"""Persona management for multi-wake-word support.

This module provides persona loading and management for the voice pipeline.
Personas are discovered from markdown files with YAML frontmatter.

Features:
    - Auto-discovery from prompts/personas/*.md
    - YAML frontmatter parsing for wake_words, voice
    - Persona lookup by name or wake word
    - Case-insensitive wake word matching
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog


@dataclass
class Persona:
    """
    A persona with wake words and voice settings.

    Attributes:
        name: Persona name (e.g., "Jarvis", "Motoko")
        wake_words: List of wake word phrases (e.g., ["hey jarvis", "jarvis"])
        voice: TTS voice name (e.g., "echo", "nova", "onyx")
        prompt: The full persona prompt (markdown content)
    """

    name: str
    wake_words: list[str] = field(default_factory=list)
    voice: str = "echo"
    prompt: str = ""


class PersonaManager:
    """
    Manages multiple personas with auto-discovery from files.

    Personas are discovered from prompts/personas/*.md with YAML frontmatter.
    Each file should have:

    ```yaml
    ---
    name: PersonaName
    wake_words:
      - hey persona
      - persona
    voice: echo
    ---
    # Persona prompt content here...
    ```

    Examples:
        >>> manager = PersonaManager()
        >>> jarvis = manager.get_persona("jarvis")
        >>> print(jarvis.voice)  # 'echo'
        >>> persona = manager.get_persona_by_wake_word("hey jarvis")
        >>> print(persona.name)  # 'Jarvis'
    """

    def __init__(
        self,
        personas_dir: str | Path = "prompts/personas",
        auto_discover: bool = True,
    ):
        """
        Initialize persona manager.

        Args:
            personas_dir: Directory containing persona files
            auto_discover: Whether to auto-load personas from directory
        """
        self._personas: dict[str, Persona] = {}
        self._wake_word_map: dict[str, str] = {}  # wake_word -> persona_name
        self._current: str | None = None
        self._log = structlog.get_logger("voice.persona")

        if auto_discover:
            self._load_personas(Path(personas_dir))

    def _load_personas(self, personas_dir: Path) -> None:
        """
        Auto-discover and load personas from directory.

        Args:
            personas_dir: Directory to scan for *.md files
        """
        if not personas_dir.exists():
            self._log.warning("personas_dir_not_found", path=str(personas_dir))
            return

        for path in personas_dir.glob("*.md"):
            try:
                persona = self._parse_persona(path)
                self._personas[persona.name.lower()] = persona

                # Build wake word lookup map
                for wake_word in persona.wake_words:
                    self._wake_word_map[wake_word.lower()] = persona.name.lower()

                self._log.debug(
                    "persona_loaded",
                    name=persona.name,
                    wake_words=persona.wake_words,
                    voice=persona.voice,
                )

            except Exception as e:
                self._log.error("persona_load_error", path=str(path), error=str(e))

        self._log.info(
            "personas_loaded",
            count=len(self._personas),
            names=list(self._personas.keys()),
        )

    def _parse_persona(self, path: Path) -> Persona:
        """
        Parse persona from markdown with YAML frontmatter.

        Args:
            path: Path to persona markdown file

        Returns:
            Parsed Persona object

        Raises:
            ValueError: If file format is invalid
        """
        content = path.read_text(encoding="utf-8")

        # Split frontmatter and content
        if not content.startswith("---"):
            raise ValueError(f"No YAML frontmatter in {path}")

        # Find end of frontmatter
        end_idx = content.find("---", 3)
        if end_idx == -1:
            raise ValueError(f"Unclosed YAML frontmatter in {path}")

        frontmatter_str = content[3:end_idx].strip()
        markdown_content = content[end_idx + 3 :].strip()

        # Parse YAML
        import yaml

        frontmatter: dict[str, Any] = yaml.safe_load(frontmatter_str) or {}

        # Extract fields
        name = frontmatter.get("name", path.stem.title())

        # Handle both wake_word (string) and wake_words (list) formats
        wake_words: list[str] = []
        if "wake_words" in frontmatter:
            ww = frontmatter["wake_words"]
            if isinstance(ww, list):
                wake_words = ww
            else:
                wake_words = [str(ww)]
        elif "wake_word" in frontmatter:
            # Legacy single wake_word support
            wake_words = [str(frontmatter["wake_word"])]

        voice = frontmatter.get("voice", "echo")

        return Persona(
            name=name,
            wake_words=wake_words,
            voice=voice,
            prompt=markdown_content,
        )

    def get_persona(self, name: str) -> Persona | None:
        """
        Get persona by name.

        Args:
            name: Persona name (case-insensitive)

        Returns:
            Persona object or None if not found
        """
        return self._personas.get(name.lower())

    def get_persona_by_wake_word(self, wake_word: str) -> Persona | None:
        """
        Get persona that matches the given wake word.

        Args:
            wake_word: Wake word phrase (case-insensitive)

        Returns:
            Persona object or None if not found
        """
        persona_name = self._wake_word_map.get(wake_word.lower())
        if persona_name:
            return self._personas.get(persona_name)
        return None

    def get_all_personas(self) -> list[Persona]:
        """
        Get all loaded personas.

        Returns:
            List of all Persona objects
        """
        return list(self._personas.values())

    def get_wake_words(self) -> list[str]:
        """
        Get all registered wake words.

        Returns:
            List of all wake word phrases
        """
        return list(self._wake_word_map.keys())

    def get_voice(self, persona_name: str) -> str:
        """
        Get TTS voice for a persona.

        Args:
            persona_name: Name of persona

        Returns:
            Voice name for TTS, defaults to "echo"
        """
        persona = self.get_persona(persona_name)
        if persona:
            return persona.voice
        return "echo"

    def get_current(self) -> Persona | None:
        """
        Get currently active persona.

        Returns:
            Current Persona or None
        """
        if self._current:
            return self._personas.get(self._current)
        return None

    def set_current(self, name: str) -> bool:
        """
        Set the current active persona.

        Args:
            name: Persona name to activate

        Returns:
            True if persona found and activated, False otherwise
        """
        if name.lower() in self._personas:
            self._current = name.lower()
            self._log.info("persona_switched", name=name)
            return True
        return False

    async def switch_to(self, persona: str) -> None:
        """
        Switch active persona (async interface).

        Args:
            persona: Name of persona to switch to
        """
        self.set_current(persona)

    def add_persona(self, persona: Persona) -> None:
        """
        Add a persona programmatically.

        Args:
            persona: Persona object to add
        """
        self._personas[persona.name.lower()] = persona
        for wake_word in persona.wake_words:
            self._wake_word_map[wake_word.lower()] = persona.name.lower()
