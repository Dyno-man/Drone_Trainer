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

        # Draw v2 obstacles (cylinders, spheres) behind drones
        for obs in state.get("obstacles", ()):
            self._draw_obstacle(obs)

        pursuer = state["pursuer_pos"]
        target = state["target_pos"]
        self.draw_polyline(state.get("pursuer_trail", []), (64, 180, 255))
        self.draw_polyline(state.get("target_trail", []), (255, 163, 77))
        pg.draw.line(self.screen, (190, 198, 208), self.project(pursuer), self.project(target), 1)
        self._draw_viewport(state)

        # v2: draw explicit drone body for agent when yaw/body params provided
        yaw = state.get("yaw")
        if yaw is not None:
            self._draw_drone_body(pursuer, (54, 162, 235), yaw,
                                  state.get("body_radius", 0.18),
                                  state.get("body_height", 0.45),
                                  state.get("rotor_span_radius", 0.65))
        else:
            self._draw_drone(pursuer, (54, 162, 235), 7)
        self._draw_drone(target, (245, 126, 43), 7)

        if state.get("flythrough_intercept", False):
            pg.draw.circle(self.screen, (124, 255, 178), self.project(target), 18, 3)
        self._draw_hud(state)

        if self.render_mode == "human":
            pg.display.flip()
            self.clock.tick(30)
            return None
        array = pg.surfarray.array3d(self.screen)
        return np.transpose(array, (1, 0, 2)).copy()

    # ------------------------------------------------------------------
    # v1 helpers (unchanged)
    # ------------------------------------------------------------------

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

    def _draw_viewport(self, state: dict[str, Any]) -> None:
        pg = self.pygame
        pursuer = np.asarray(state["pursuer_pos"], dtype=np.float32)
        heading = np.asarray(state.get("pursuer_heading", np.array([1.0, 0.0, 0.0])), dtype=np.float32)
        heading_norm = float(np.linalg.norm(heading))
        if heading_norm <= 1e-8:
            return
        heading = heading / heading_norm
        viewport_range = float(state.get("viewport_range", 0.0))
        if viewport_range <= 0.0:
            return
        ray_end = pursuer + heading * min(viewport_range, self.world_xy * 0.9)
        color = (97, 214, 164) if state.get("target_visible", False) else (112, 128, 150)
        pg.draw.line(self.screen, color, self.project(pursuer), self.project(ray_end), 2)

        half_angle = np.deg2rad(float(state.get("horizontal_fov_deg", 60.0)) * 0.5)
        yaw = float(np.arctan2(heading[1], heading[0]))
        for side in (-1.0, 1.0):
            direction = np.array(
                [np.cos(yaw + side * half_angle), np.sin(yaw + side * half_angle), heading[2]],
                dtype=np.float32,
            )
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm > 1e-8:
                direction /= direction_norm
                edge = pursuer + direction * min(viewport_range, self.world_xy * 0.65)
                pg.draw.line(self.screen, (74, 96, 120), self.project(pursuer), self.project(edge), 1)

    def _draw_hud(self, state: dict[str, Any]) -> None:
        lines = [
            f"step {state['step_count']} / {state['max_steps']}",
            f"distance {state['distance']:.2f}",
            f"reward {state.get('reward', 0.0):.2f}",
            f"visible {state.get('target_visible', False)} lock {state.get('has_target_lock', False)} unseen {state.get('steps_since_seen', 0)}",
            f"flythrough {state.get('flythrough_intercept', False)} captured {state.get('captured', False)}",
            f"crashed {state.get('crashed', False)} oob {state.get('out_of_bounds', False)}",
        ]
        for idx, text in enumerate(lines):
            image = self.font.render(text, True, (232, 236, 241))
            self.screen.blit(image, (12, 12 + idx * 20))

    def close(self) -> None:
        self.pygame.display.quit()
        self.pygame.quit()

    # ------------------------------------------------------------------
    # v2 extras (stateless — all args from state dict)
    # ------------------------------------------------------------------

    def _draw_obstacle(self, obs: dict[str, Any]) -> None:
        """Dispatch to cylinder or sphere drawing based on obstacle type."""
        if obs.get("type") == "sphere":
            self._draw_sphere(obs)
        else:
            self._draw_cylinder(obs)

    def _draw_cylinder(self, obs: dict[str, Any]) -> None:
        """Draw a vertical cylinder as two wireframe circles (base + top) + vertical edges."""
        center = np.asarray(obs["center"], dtype=np.float32)
        radius = obs["radius"]
        height = obs["height"]
        z_min = float(center[2])       # cylinder base
        z_max = z_min + height          # cylinder top
        color = (100, 100, 110)
        segments = 16

        def _circle_projected(z: float) -> list[tuple[int, int]]:
            pts: list[tuple[int, int]] = []
            for i in range(segments):
                a = 2.0 * np.pi * i / segments
                x = center[0] + radius * np.cos(a)
                y = center[1] + radius * np.sin(a)
                pts.append(self.project(np.array([x, y, z], dtype=np.float32)))
            return pts

        base_pts = _circle_projected(z_min)
        top_pts = _circle_projected(z_max)

        if len(base_pts) >= 2:
            self.pygame.draw.lines(self.screen, color, True, base_pts, 1)
        if len(top_pts) >= 2:
            self.pygame.draw.lines(self.screen, color, True, top_pts, 1)
        # Vertical edges (every 4th for readability)
        for i in range(0, segments, 4):
            self.pygame.draw.line(self.screen, color, base_pts[i], top_pts[i], 1)

    def _draw_sphere(self, obs: dict[str, Any]) -> None:
        """Draw a sphere as a wireframe circle in screen space."""
        center = np.asarray(obs["center"], dtype=np.float32)
        radius = obs["radius"]
        color = (100, 100, 110)
        proj = self.project(center)
        scale = min(self.width, self.height) / (self.world_xy * 3.0)
        # Compute screen-space radius via x-offset projection
        screen_r = int(
            abs(
                self.project(np.array([center[0] + radius, center[1], center[2]], dtype=np.float32))[0]
                - proj[0]
            )
        )
        self.pygame.draw.circle(self.screen, color, proj, max(1, screen_r), 1)

    def _draw_drone_body(
        self,
        position: np.ndarray,
        color: tuple[int, int, int],
        yaw: float,
        body_radius: float,
        _body_height: float,
        rotor_span_radius: float,
    ) -> None:
        """Draw a quadcopter: body + 4 rotor arms + 4 rotors, oriented by yaw.

        Drawing is done in screen space; yaw=0 means the drone points along +x
        (right on screen), yaw=pi/2 means +y (up on screen).
        """
        pg = self.pygame
        screen_pos = self.project(position)
        scale = min(self.width, self.height) / (self.world_xy * 3.0)

        body_size = int(max(3, body_radius * 2.0 * scale * 0.7))
        arm_span = int(max(4, rotor_span_radius * scale * 0.6))

        cy, sy = np.cos(yaw), np.sin(yaw)
        # arm_dirs: front(+x), right(+y), back(-x), left(-y) in drone frame
        arm_dirs = [
            (cy, sy),       # front
            (-sy, cy),      # right
            (-cy, -sy),     # back
            (sy, -cy),      # left
        ]

        arm_color = (180, 185, 190)
        for dx, dy in arm_dirs:
            x1 = int(screen_pos[0] + dx * arm_span * 0.4)
            y1 = int(screen_pos[1] - dy * arm_span * 0.4)
            x2 = int(screen_pos[0] + dx * arm_span)
            y2 = int(screen_pos[1] - dy * arm_span)
            pg.draw.line(self.screen, arm_color, (x1, y1), (x2, y2), 2)

        rotor_color = (200, 205, 210)
        rotor_r = max(2, arm_span // 4)
        for dx, dy in arm_dirs:
            rx = int(screen_pos[0] + dx * arm_span)
            ry = int(screen_pos[1] - dy * arm_span)
            pg.draw.circle(self.screen, rotor_color, (rx, ry), rotor_r)
            pg.draw.circle(self.screen, (50, 50, 55), (rx, ry), rotor_r, 1)

        body_r = max(2, body_size // 2)
        pg.draw.circle(self.screen, color, screen_pos, body_size)
        pg.draw.circle(self.screen, (235, 240, 245), screen_pos, body_size, 1)
