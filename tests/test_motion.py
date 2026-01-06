"""Tests for motion control."""

import pytest


class TestBlendController:
    """Tests for BlendController."""

    @pytest.mark.asyncio
    async def test_initialization(self, mock_robot):
        """Test blend controller initialization."""
        from reachy_agent.motion.controller import BlendController

        controller = BlendController(mock_robot)
        assert controller is not None

    # TODO: Add more tests
    # - test_primary_source
    # - test_overlay_blending
    # - test_pose_clamping
    # - test_30hz_timing
