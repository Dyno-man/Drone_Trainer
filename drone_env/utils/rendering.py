from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DroneRenderer:
    width: int
    height: int
    world_xy: float
    world_z: float
    render_mode: str

    def __post_init__(self) -> None:
        if self.render_mode == "rgb_array":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame

        self.pygame = pygame
        pygame.init()
        pygame.font.init()
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("DroneIntercept3D-v0")
        else:
            self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 16)

    def project(self, point: np.ndarray) -> tuple[int, int]:
        x, y, z = np.asarray(point, dtype=np.float32)
        scale = min(self.width, self.height) / (self.world_xy * 3.0)
        sx = self.width / 2 + (x - y) * 0.866 * scale
        sy = self.height * 0.70 + (x + y) * 0.35 * scale - z * 1.7 * scale
        return int(sx), int(sy)

    def draw_polyline(self, points: list[np.ndarray], color: tuple[int, int, int]) -> None:
        if len(points) < 2:
            return
        projected = [self.project(point) for point in points]
        self.pygame.draw.lines(self.screen, color, False, projected, 2)

    def draw(self, state: dict[str, Any]) -> np.ndarray | None:
        pg = self.pygame
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return None

        self.screen.fill((14, 18, 24))
        self._draw_floor()
        self._draw_box()

        pursuer = state["pursuer_pos"]
        target = state["target_pos"]
        self.draw_polyline(state.get("pursuer_trail", []), (64, 180, 255))
        self.draw_polyline(state.get("target_trail", []), (255, 163, 77))
        pg.draw.line(self.screen, (190, 198, 208), self.project(pursuer), self.project(target), 1)
        self._draw_drone(pursuer, (54, 162, 235), 7)
        self._draw_drone(target, (245, 126, 43), 7)
        self._draw_hud(state)

        if self.render_mode == "human":
            pg.display.flip()
            self.clock.tick(30)
            return None
        array = pg.surfarray.array3d(self.screen)
        return np.transpose(array, (1, 0, 2)).copy()

    def _draw_floor(self) -> None:
        pg = self.pygame
        color = (45, 55, 68)
        step = 25
        for value in range(int(-self.world_xy), int(self.world_xy) + 1, step):
            pg.draw.line(
                self.screen,
                color,
                self.project(np.array([value, -self.world_xy, 0], dtype=np.float32)),
                self.project(np.array([value, self.world_xy, 0], dtype=np.float32)),
                1,
            )
            pg.draw.line(
                self.screen,
                color,
                self.project(np.array([-self.world_xy, value, 0], dtype=np.float32)),
                self.project(np.array([self.world_xy, value, 0], dtype=np.float32)),
                1,
            )

    def _draw_box(self) -> None:
        pg = self.pygame
        color = (88, 102, 122)
        x = self.world_xy
        z = self.world_z
        corners = [
            np.array([sx * x, sy * x, sz * z], dtype=np.float32)
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (0, 1)
        ]
        for a in corners:
            for b in corners:
                diff = np.count_nonzero(np.abs(a - b) > 1e-5)
                if diff == 1:
                    pg.draw.line(self.screen, color, self.project(a), self.project(b), 1)

    def _draw_drone(self, position: np.ndarray, color: tuple[int, int, int], radius: int) -> None:
        pg = self.pygame
        center = self.project(position)
        pg.draw.circle(self.screen, color, center, radius)
        pg.draw.circle(self.screen, (235, 240, 245), center, radius, 1)

    def _draw_hud(self, state: dict[str, Any]) -> None:
        lines = [
            f"step {state['step_count']} / {state['max_steps']}",
            f"distance {state['distance']:.2f}",
            f"reward {state.get('reward', 0.0):.2f}",
            f"captured {state.get('captured', False)} crashed {state.get('crashed', False)} oob {state.get('out_of_bounds', False)}",
        ]
        for idx, text in enumerate(lines):
            image = self.font.render(text, True, (232, 236, 241))
            self.screen.blit(image, (12, 12 + idx * 20))

    def close(self) -> None:
        self.pygame.display.quit()
        self.pygame.quit()
