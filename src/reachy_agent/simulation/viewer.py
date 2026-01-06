"""Real-time 3D visualization for MuJoCo simulation.

This module provides a visualization interface for the Reachy Mini simulation.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import structlog

try:
    import mujoco
    from mujoco import viewer as mj_viewer

    MUJOCO_AVAILABLE = True
except ImportError:
    mujoco = None  # type: ignore
    mj_viewer = None  # type: ignore
    MUJOCO_AVAILABLE = False

logger = structlog.get_logger()


class SimulationViewer:
    """
    Real-time 3D visualization for MuJoCo simulation.

    Provides an interactive viewer window with camera controls,
    debug overlays, and recording capabilities.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        title: Window title

    Example:
        >>> viewer = SimulationViewer(model, data)
        >>> viewer.start()
        >>> # ... simulation runs ...
        >>> viewer.stop()
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        title: str = "Reachy Mini Simulation",
    ) -> None:
        """Initialize the viewer."""
        self._model = model
        self._data = data
        self._title = title
        self._viewer: Any = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._log = logger.bind(component="sim_viewer")

        # Overlay settings
        self._show_joints = False
        self._show_contacts = False
        self._show_forces = False

        # Recording
        self._recording = False
        self._frames: list[np.ndarray] = []

    def start(self) -> None:
        """Start the viewer in a background thread."""
        if not MUJOCO_AVAILABLE:
            self._log.warning("viewer_not_available", reason="mujoco_not_installed")
            return

        if self._running:
            return

        self._log.info("starting_viewer")
        self._running = True
        self._viewer = mj_viewer.launch_passive(
            self._model,
            self._data,
            show_left_ui=True,
            show_right_ui=True,
        )

    def stop(self) -> None:
        """Stop the viewer."""
        if not self._running:
            return

        self._log.info("stopping_viewer")
        self._running = False

        if self._viewer:
            self._viewer.close()
            self._viewer = None

    def sync(self) -> None:
        """Sync viewer with simulation state."""
        if self._viewer and self._running:
            self._viewer.sync()

            # Capture frame if recording
            if self._recording:
                self._capture_frame()

    def is_running(self) -> bool:
        """Check if viewer is running."""
        if self._viewer:
            return self._viewer.is_running()
        return False

    # Camera controls

    def set_camera(
        self,
        azimuth: float | None = None,
        elevation: float | None = None,
        distance: float | None = None,
        lookat: tuple[float, float, float] | None = None,
    ) -> None:
        """Set camera position and orientation.

        Args:
            azimuth: Horizontal angle in degrees
            elevation: Vertical angle in degrees
            distance: Distance from lookat point
            lookat: Point to look at (x, y, z)
        """
        if not self._viewer:
            return

        cam = self._viewer.cam
        if azimuth is not None:
            cam.azimuth = azimuth
        if elevation is not None:
            cam.elevation = elevation
        if distance is not None:
            cam.distance = distance
        if lookat is not None:
            cam.lookat[:] = lookat

    def set_camera_preset(self, preset: str) -> None:
        """Set camera to a preset position.

        Args:
            preset: One of "front", "side", "top", "perspective"
        """
        presets = {
            "front": {"azimuth": 180, "elevation": -20, "distance": 1.5},
            "side": {"azimuth": 90, "elevation": -20, "distance": 1.5},
            "top": {"azimuth": 180, "elevation": -90, "distance": 2.0},
            "perspective": {"azimuth": 135, "elevation": -30, "distance": 1.8},
        }

        if preset in presets:
            self.set_camera(**presets[preset])

    # Debug overlays

    def toggle_joint_overlay(self) -> None:
        """Toggle joint visualization overlay."""
        self._show_joints = not self._show_joints
        if self._viewer:
            self._viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = self._show_joints

    def toggle_contact_overlay(self) -> None:
        """Toggle contact point visualization."""
        self._show_contacts = not self._show_contacts
        if self._viewer:
            self._viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = (
                self._show_contacts
            )
            self._viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = (
                self._show_contacts
            )

    def toggle_force_overlay(self) -> None:
        """Toggle force visualization."""
        self._show_forces = not self._show_forces
        if self._viewer:
            self._viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = (
                self._show_forces
            )

    # Recording

    def start_recording(self) -> None:
        """Start recording frames."""
        self._log.info("starting_recording")
        self._recording = True
        self._frames = []

    def stop_recording(self) -> list[np.ndarray]:
        """Stop recording and return frames.

        Returns:
            List of recorded frames as numpy arrays
        """
        self._log.info("stopping_recording", frame_count=len(self._frames))
        self._recording = False
        frames = self._frames
        self._frames = []
        return frames

    def _capture_frame(self) -> None:
        """Capture current frame."""
        if not self._viewer or not MUJOCO_AVAILABLE:
            return

        # Use MuJoCo's renderer to capture
        try:
            width, height = 1280, 720
            renderer = mujoco.Renderer(self._model, height, width)
            renderer.update_scene(self._data, camera=-1)
            pixels = renderer.render()
            self._frames.append(pixels.copy())
        except Exception as e:
            self._log.error("frame_capture_failed", error=str(e))

    def save_video(
        self,
        path: str,
        fps: int = 30,
        frames: list[np.ndarray] | None = None,
    ) -> None:
        """Save frames to video file.

        Args:
            path: Output video path (e.g., "output.mp4")
            fps: Frames per second
            frames: Frames to save (uses recorded if None)
        """
        frames_to_save = frames or self._frames

        if not frames_to_save:
            self._log.warning("no_frames_to_save")
            return

        self._log.info("saving_video", path=path, frames=len(frames_to_save), fps=fps)

        try:
            import imageio

            writer = imageio.get_writer(path, fps=fps)
            for frame in frames_to_save:
                writer.append_data(frame)
            writer.close()
            self._log.info("video_saved", path=path)
        except ImportError:
            self._log.error("imageio_not_installed")
        except Exception as e:
            self._log.error("video_save_failed", error=str(e))

    def take_screenshot(self, path: str) -> None:
        """Save current view as image.

        Args:
            path: Output image path (e.g., "screenshot.png")
        """
        if not MUJOCO_AVAILABLE or not self._model:
            self._log.warning("screenshot_not_available")
            return

        self._log.info("taking_screenshot", path=path)

        try:
            width, height = 1920, 1080
            renderer = mujoco.Renderer(self._model, height, width)
            renderer.update_scene(self._data, camera=-1)
            pixels = renderer.render()

            import imageio

            imageio.imwrite(path, pixels)
            self._log.info("screenshot_saved", path=path)
        except Exception as e:
            self._log.error("screenshot_failed", error=str(e))


class HeadlessRenderer:
    """
    Headless renderer for CI/CD and testing.

    Renders to offscreen buffer without requiring a display.
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        width: int = 640,
        height: int = 480,
    ) -> None:
        """Initialize headless renderer."""
        self._model = model
        self._data = data
        self._width = width
        self._height = height
        self._renderer: Any = None
        self._log = logger.bind(component="headless_renderer")

    def __enter__(self) -> HeadlessRenderer:
        """Enter context manager."""
        if MUJOCO_AVAILABLE and self._model:
            self._renderer = mujoco.Renderer(self._model, self._height, self._width)
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context manager."""
        self._renderer = None

    def render(self, camera: int | str = -1) -> np.ndarray:
        """Render scene from camera.

        Args:
            camera: Camera ID or name (-1 for free camera)

        Returns:
            RGB pixel array
        """
        if not self._renderer:
            return np.zeros((self._height, self._width, 3), dtype=np.uint8)

        if isinstance(camera, str):
            camera = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera)

        self._renderer.update_scene(self._data, camera=camera)
        return self._renderer.render()

    def render_depth(self, camera: int | str = -1) -> np.ndarray:
        """Render depth map from camera.

        Args:
            camera: Camera ID or name

        Returns:
            Depth array (meters)
        """
        if not self._renderer:
            return np.zeros((self._height, self._width), dtype=np.float32)

        if isinstance(camera, str):
            camera = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera)

        self._renderer.update_scene(self._data, camera=camera)
        self._renderer.enable_depth_rendering(True)
        depth = self._renderer.render()
        self._renderer.enable_depth_rendering(False)
        return depth
