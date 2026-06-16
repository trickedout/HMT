from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ScreenshotGroundingRequest:
    image_path: Path
    semantic_description: str


class ScreenshotGrounder:
    """Interface stub for visual grounding.

    Local checks intentionally do not require a heavy vision model.
    """

    def ground(self, request: ScreenshotGroundingRequest) -> tuple[float, float] | None:
        raise NotImplementedError("Configure a visual grounding model for screenshot-based environments.")
