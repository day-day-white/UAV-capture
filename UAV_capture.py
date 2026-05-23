# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:11:38 2026

@author: wanwu
"""
# -*- coding: utf-8 -*-
"""
UAV Swarm Capture Simulation with DeepSeek API Integration
- 在搜索阶段调用 DeepSeek API 生成无人机速度指令
- 增加 API 调用间隔控制，避免过度请求
- 失败时自动降级到原有 PSO 逻辑
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Patch
from matplotlib.lines import Line2D
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import seaborn as sns
from scipy.ndimage import convolve
from scipy.optimize import linear_sum_assignment
import math
import requests  # 新增：用于调用 API
import json     # 新增：处理 JSON 数据

# ---------------------------- 全局配置（新增 API 相关参数）----------------------------
@dataclass
class Config:
    """存储所有模拟参数（新增 API 配置）"""
    # 基础网格/时间参数
    grid_width: int = 100
    grid_height: int = 100
    cell_size: float = 1.0
    dt: float = 0.1
    max_steps: int = 1000

    # 无人机参数
    num_uavs: int = 6
    uav_speed: float = 3.0
    uav_sensor_range: float = 15.0
    uav_avoid_range: float = 3.0
    uav_obstacle_avoid_weight: float = 0.8

    # 目标参数
    num_targets: int = 8
    target_speed: float = 1.5
    target_initial_pos: List[Tuple[float, float]] = field(default_factory=lambda: [(30,30), (70,70), (50,20), (20,80)])
    target_escape_strength: float = 1.2

    # 元胞自动机参数
    pheromone_decay: float = 0.97
    pheromone_deposit: float = 3.0
    pheromone_diffusion_sigma: float = 1.0

    # PSO参数
    pso_w_min: float = 0.6
    pso_w_max: float = 0.9
    pso_c1_min: float = 2.0
    pso_c1_max: float = 2.5
    pso_c2_min: float = 1.5
    pso_c2_max: float = 2.0

    # 围捕核心参数
    capture_distance: float = 20.0
    capture_radius: float = 12.0
    capture_success_distance: float = 10.0
    capture_success_frames: int = 20
    nn_input_dim: int = 4
    nn_hidden_dim: int = 16
    nn_output_dim: int = 2

    # 障碍物参数
    obstacle_density: float = 0.05
    obstacle_value: float = 1.0

    # 可视化参数
    plot_interval: int = 1
    trail_length: int = 30
    show_sensor_range: bool = True
    show_trails: bool = True
    pheromone_alpha: float = 0.7
    pheromone_cmap: str = 'Reds'
    obstacle_color: str = 'black'
    obstacle_alpha: float = 0.8
    uav_search_color: str = 'dodgerblue'
    uav_capture_color: str = 'red'
    uav_edgecolor: str = 'white'
    target_colors: List[str] = field(default_factory=lambda: ['red', 'orange', 'green', 'purple', 'brown'])
    target_marker: str = '*'
    target_edgecolor: str = 'gold'
    trail_alpha: float = 0.3
    show_grid: bool = True

    # 同伴排斥参数
    crowd_avoid_range: float = 20.0
    crowd_avoid_weight_min: float = 0.1
    crowd_avoid_weight_max: float = 0.4
    crowd_density_threshold: float = 0.3

    # 智能抓捕参数
    target_priority_speed_weight: float = 0.6
    target_priority_capture_weight: float = 0.4
    dwa_v_resolution: float = 0.5
    dwa_omega_resolution: float = np.pi/8
    dwa_predict_time: float = 0.8
    dwa_to_goal_cost_weight: float = 0.8
    dwa_obstacle_cost_weight: float = 0.8
    dwa_speed_cost_weight: float = 0.1
    search_area_divide: bool = False
    search_area_margin: float = 5.0

    # ---------- 新增 DeepSeek API 配置 ----------
    use_api: bool = False                      # 是否启用 API 决策（默认关闭）
    api_key: str = "****"     # 请替换为实际密钥
    api_url: str = "https://api.deepseek.com/v1/chat/completions"
    api_call_interval: int = 10                 # 每多少步调用一次 API
    api_timeout: float = 2.0                    # 请求超时时间（秒）
    # -----------------------------------------

# ---------------------------- 元胞自动机网格（保持不变）----------------------------
class CAGrid:
    # ... 原有代码保持不变 ...
    def __init__(self, config: Config):
        self.config = config
        self.width = config.grid_width
        self.height = config.grid_height
        self.pheromone = np.zeros((self.height, self.width))
        self.searched = np.zeros((self.height, self.width), dtype=bool)
        self.obstacle = np.zeros((self.height, self.width), dtype=bool)
        self._init_obstacles()

    def _init_obstacles(self):
        """随机生成障碍物，并确保目标/无人机初始位置不是障碍"""
        mask = np.random.rand(self.height, self.width) < self.config.obstacle_density
        self.obstacle[mask] = True
        # 确保目标初始位置无障碍物
        for pos in self.config.target_initial_pos:
            ix, iy = int(pos[0]), int(pos[1])
            if 0 <= ix < self.width and 0 <= iy < self.height:
                self.obstacle[iy, ix] = False

    def step(self):
        self.pheromone *= self.config.pheromone_decay
        if self.config.pheromone_diffusion_sigma > 0:
            self._diffuse_gaussian()

    def _diffuse_gaussian(self):
        kernel = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=float) / 16.0
        self.pheromone = convolve(self.pheromone, kernel, mode='constant', cval=0.0)

    def deposit_pheromone(self, pos: Tuple[float, float], amount: float = None):
        x, y = int(pos[0]), int(pos[1])
        if 0 <= x < self.width and 0 <= y < self.height and not self.obstacle[y, x]:
            amount = amount or self.config.pheromone_deposit
            self.pheromone[y, x] += amount

    def get_pheromone_at(self, pos: Tuple[float, float]) -> float:
        x, y = int(pos[0]), int(pos[1])
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.pheromone[y, x]
        return 0.0

    def mark_searched(self, pos: Tuple[float, float]):
        x, y = int(pos[0]), int(pos[1])
        if 0 <= x < self.width and 0 <= y < self.height:
            self.searched[y, x] = True

    def is_obstacle(self, pos: Tuple[float, float]) -> bool:
        x, y = int(pos[0]), int(pos[1])
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.obstacle[y, x]
        return True  # 边界视为障碍物

    def get_obstacle_positions(self) -> np.ndarray:
        y, x = np.where(self.obstacle)
        return np.column_stack((x, y))
    
    def get_local_density(self, pos: Tuple[float, float], radius: float) -> float:
        """计算指定位置周围的障碍物密度"""
        x, y = int(pos[0]), int(pos[1])
        r = int(radius)
        x_min = max(0, x - r)
        x_max = min(self.width, x + r)
        y_min = max(0, y - r)
        y_max = min(self.height, y + r)
        area = (x_max - x_min) * (y_max - y_min)
        if area == 0:
            return 0.0
        obstacle_count = np.sum(self.obstacle[y_min:y_max, x_min:x_max])
        return obstacle_count / area

# ---------------------------- 无人机类（保持不变）----------------------------
class UAV:
    # ... 原有代码保持不变 ...
    def __init__(self, uav_id: int, init_pos: Tuple[float, float], config: Config):
        self.id = uav_id
        self.pos = np.array(init_pos, dtype=float)
        self.vel = (np.random.randn(2) * 0.5 + np.array([1.0, 1.0])) * 0.5
        self.config = config
        self.best_pos = self.pos.copy()
        self.best_fitness = -np.inf
        self.mode = 'search'
        self.target_visible = False
        self.trail = np.zeros((config.trail_length, 2))
        self.search_area = self._init_search_area()
        self.omega = 0.0
        self.max_omega = np.pi/4

    def _init_search_area(self) -> Tuple[float, float, float, float]:
        if not self.config.search_area_divide:
            return (0, 0, self.config.grid_width, self.config.grid_height)
        col_width = self.config.grid_width / self.config.num_uavs
        x_min = max(0, self.id * col_width - self.config.search_area_margin)
        x_max = min(self.config.grid_width, (self.id + 1) * col_width + self.config.search_area_margin)
        y_min = 0
        y_max = self.config.grid_height
        return (x_min, y_min, x_max, y_max)

    def sense_pheromone(self, grid: CAGrid) -> float:
        return grid.get_pheromone_at(self.pos)

    def update_best(self, fitness: float):
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_pos = self.pos.copy()

    def get_adaptive_pso_params(self, grid: CAGrid, other_uavs: List['UAV']) -> Tuple[float, float, float]:
        local_pheromone = self._get_local_pheromone(grid, radius=10)
        pheromone_norm = np.clip(local_pheromone / 10.0, 0, 1)
        w = self.config.pso_w_max - (self.config.pso_w_max - self.config.pso_w_min) * pheromone_norm

        local_uav_density = self._get_local_uav_density(other_uavs, radius=20)
        c1 = self.config.pso_c1_min + (self.config.pso_c1_max - self.config.pso_c1_min) * local_uav_density

        searched_ratio = self._get_searched_ratio(grid)
        c2 = self.config.pso_c2_max - (self.config.pso_c2_max - self.config.pso_c2_min) * searched_ratio

        return w, c1, c2

    def _get_local_pheromone(self, grid: CAGrid, radius: float) -> float:
        x, y = int(self.pos[0]), int(self.pos[1])
        r = int(radius)
        x_min = max(0, x - r)
        x_max = min(grid.width, x + r)
        y_min = max(0, y - r)
        y_max = min(grid.height, y + r)
        return np.mean(grid.pheromone[y_min:y_max, x_min:x_max])

    def _get_local_uav_density(self, other_uavs: List['UAV'], radius: float) -> float:
        count = 0
        for other in other_uavs:
            if other.id == self.id:
                continue
            dist = np.linalg.norm(self.pos - other.pos)
            if dist < radius:
                count += 1
        max_possible = self.config.num_uavs - 1
        return np.clip(count / max_possible if max_possible > 0 else 0, 0, 1)

    def _get_searched_ratio(self, grid: CAGrid) -> float:
        x_min, y_min, x_max, y_max = self.search_area
        x_min, x_max = int(x_min), int(x_max)
        y_min, y_max = int(y_min), int(y_max)
        searched = grid.searched[y_min:y_max, x_min:x_max]
        total = searched.size
        if total == 0:
            return 0.0
        return np.sum(searched) / total

    def pso_update(self, global_best_pos: np.ndarray, grid: CAGrid, other_uavs: List['UAV']):
        w, c1, c2 = self.get_adaptive_pso_params(grid, other_uavs)
        r1, r2 = np.random.rand(2) * 1.2
        
        cognitive = c1 * r1 * (self.best_pos - self.pos)
        social = c2 * r2 * (global_best_pos - self.pos)
        self.vel = w * self.vel + cognitive + social + (np.random.randn(2) * 0.1)
        
        speed = np.linalg.norm(self.vel)
        if speed > self.config.uav_speed:
            self.vel = self.vel / speed * self.config.uav_speed
        elif speed < 0.5:
            self.vel = self.vel / (speed + 1e-6) * 0.5
        
        new_pos = self.pos + self.vel * self.config.dt
        new_pos = self._simple_avoid_obstacle(new_pos, grid)
        new_pos = self._avoid_boundary(new_pos)
        
        self.pos = new_pos
        grid.mark_searched(self.pos)

    def _simple_avoid_obstacle(self, new_pos: np.ndarray, grid: CAGrid) -> np.ndarray:
        if not grid.is_obstacle(new_pos):
            return new_pos
        
        directions = [np.array([dx, dy]) for dx in [-1,0,1] for dy in [-1,0,1] if dx !=0 or dy !=0]
        for dir in directions:
            test_pos = self.pos + dir * self.config.uav_speed * self.config.dt
            if not grid.is_obstacle(test_pos):
                return test_pos
        
        return self.pos + (np.random.randn(2) * 0.1)

    def dwa_control(self, goal_pos: np.ndarray, grid: CAGrid, target_pos: np.ndarray) -> np.ndarray:
        v_min = max(0, np.linalg.norm(self.vel) - self.config.dwa_v_resolution * 3)
        v_max = min(self.config.uav_speed, np.linalg.norm(self.vel) + self.config.dwa_v_resolution * 3)
        omega_min = max(-self.max_omega, self.omega - self.config.dwa_omega_resolution * 3)
        omega_max = min(self.max_omega, self.omega + self.config.dwa_omega_resolution * 3)

        v_candidates = np.arange(v_min, v_max + 1e-6, self.config.dwa_v_resolution)
        omega_candidates = np.arange(omega_min, omega_max + 1e-6, self.config.dwa_omega_resolution)
        if len(v_candidates) == 0:
            v_candidates = [np.linalg.norm(self.vel) + 0.1]
        if len(omega_candidates) == 0:
            omega_candidates = [self.omega + np.pi/16]

        best_score = -np.inf
        best_pos = self.pos + self.vel * self.config.dt

        for v in v_candidates:
            for omega in omega_candidates:
                traj = self._predict_trajectory(v, omega)
                end_pos = traj[-1]
                
                to_goal_cost = np.linalg.norm(end_pos - target_pos)
                obstacle_cost = 1000 if grid.is_obstacle(end_pos) else 0
                speed_cost = (self.config.uav_speed - v)
                
                total_score = (
                    -self.config.dwa_to_goal_cost_weight * to_goal_cost
                    -self.config.dwa_obstacle_cost_weight * obstacle_cost
                    -self.config.dwa_speed_cost_weight * speed_cost
                )

                if total_score > best_score:
                    best_score = total_score
                    best_pos = end_pos

        return self._avoid_boundary(best_pos)

    def _predict_trajectory(self, v: float, omega: float) -> np.ndarray:
        traj = np.zeros((int(self.config.dwa_predict_time / self.config.dt), 2))
        traj[0] = self.pos
        current_theta = np.arctan2(self.vel[1], self.vel[0]) if np.linalg.norm(self.vel) > 0 else 0

        for i in range(1, len(traj)):
            current_theta += omega * self.config.dt
            dx = v * np.cos(current_theta) * self.config.dt
            dy = v * np.sin(current_theta) * self.config.dt
            traj[i] = traj[i-1] + np.array([dx, dy])
        return traj

    def _avoid_boundary(self, new_pos: np.ndarray) -> np.ndarray:
        new_pos[0] = np.clip(new_pos[0], 1, self.config.grid_width - 2)
        new_pos[1] = np.clip(new_pos[1], 1, self.config.grid_height - 2)
        return new_pos

    def neural_control(self, target_pos: np.ndarray, target_vel: np.ndarray, nn_model, grid: CAGrid):
        rel_pos = target_pos - self.pos
        dist = np.linalg.norm(rel_pos)
        
        if dist < 1e-6:
            desired_vel = np.random.randn(2) * 0.1
        else:
            desired_vel = (rel_pos / dist) * self.config.uav_speed + (np.random.randn(2) * 0.05)
        
        speed = np.linalg.norm(desired_vel)
        if speed > self.config.uav_speed:
            desired_vel = desired_vel / speed * self.config.uav_speed
        
        self.vel = 0.8 * self.vel + 0.2 * desired_vel
        new_pos = self.pos + self.vel * self.config.dt
        new_pos = self._simple_avoid_obstacle(new_pos, grid)
        self.pos = new_pos

    def avoid_collision(self, other_uavs: List['UAV'], grid: CAGrid):
        repulsion = np.zeros(2)
        for other in other_uavs:
            if other.id == self.id:
                continue
            diff = self.pos - other.pos
            dist = np.linalg.norm(diff)
            if 0 < dist < self.config.uav_avoid_range:
                repulsion += diff / dist * (self.config.uav_avoid_range - dist) / self.config.uav_avoid_range * 0.5

        obstacle_density = grid.get_local_density(self.pos, self.config.uav_avoid_range * 2)
        obstacle_avoid_weight = self.config.uav_obstacle_avoid_weight * (1 + obstacle_density) * 0.5
        
        obstacle_repel = np.zeros(2)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                check_pos = self.pos + np.array([dx * self.config.cell_size, dy * self.config.cell_size])
                if grid.is_obstacle(check_pos):
                    diff = self.pos - check_pos
                    dist = np.linalg.norm(diff)
                    if dist > 0:
                        obstacle_repel += diff / dist * (self.config.uav_avoid_range - dist) / self.config.uav_avoid_range * 0.3

        total_repel = repulsion + obstacle_avoid_weight * obstacle_repel
        self.vel += total_repel
        
        speed = np.linalg.norm(self.vel)
        if speed > self.config.uav_speed:
            self.vel = self.vel / speed * self.config.uav_speed
        elif speed < 0.3:
            self.vel = self.vel / (speed + 1e-6) * 0.3

    def avoid_crowd(self, other_uavs: List['UAV']) -> np.ndarray:
        repulsion = np.zeros(2)
        local_density = self._get_local_uav_density(other_uavs, self.config.crowd_avoid_range)
        
        if local_density > self.config.crowd_density_threshold:
            crowd_avoid_weight = self.config.crowd_avoid_weight_max * 0.5
        else:
            crowd_avoid_weight = self.config.crowd_avoid_weight_min * 0.5

        for other in other_uavs:
            if other.id == self.id:
                continue
            diff = self.pos - other.pos
            dist = np.linalg.norm(diff)
            if 0 < dist < self.config.crowd_avoid_range:
                direction = diff / (dist + 1e-6)
                strength = (self.config.crowd_avoid_range - dist) / self.config.crowd_avoid_range
                repulsion += direction * strength * crowd_avoid_weight
        return repulsion

    def update_trail(self):
        self.trail = np.roll(self.trail, -1, axis=0)
        self.trail[-1] = self.pos

# ---------------------------- 目标类（保持不变）----------------------------
class Target:
    # ... 原有代码保持不变 ...
    def __init__(self, target_id: int, init_pos: Tuple[float, float], config: Config):
        self.id = target_id
        self.pos = np.array(init_pos, dtype=float)
        self.vel = np.random.randn(2) * 0.5 + np.array([0.5, 0.5])
        self.config = config
        self.trail = np.zeros((config.trail_length, 2))
        self.is_being_captured = False
        self.capture_counter = 0

    def step(self, uavs: List[UAV], grid: CAGrid):
        escape_dir = np.zeros(2)
        for u in uavs:
            diff = self.pos - u.pos
            dist = np.linalg.norm(diff)
            if dist > 0:
                weight = 1.0 / (dist + 0.1)
                escape_dir += diff / dist * weight

        capture_dists = [np.linalg.norm(u.pos - self.pos) for u in uavs]
        self.is_being_captured = all(d < self.config.capture_success_distance for d in capture_dists)
        if self.is_being_captured:
            self.capture_counter += 1
        else:
            self.capture_counter = 0

        if np.linalg.norm(escape_dir) > 0:
            escape_dir = escape_dir / np.linalg.norm(escape_dir)
            check_pos = self.pos + escape_dir * self.config.target_speed * self.config.dt
            if grid.is_obstacle(check_pos):
                best_dir = escape_dir
                best_cost = np.inf
                for angle in np.arange(0, 2*np.pi, np.pi/8):
                    test_dir = np.array([np.cos(angle), np.sin(angle)])
                    test_pos = self.pos + test_dir * self.config.target_speed * self.config.dt
                    if not grid.is_obstacle(test_pos):
                        dist_to_uavs = np.mean([np.linalg.norm(self.pos + test_dir - u.pos) for u in uavs])
                        cost = 1.0 / (dist_to_uavs + 1e-6)
                        if cost < best_cost:
                            best_cost = cost
                            best_dir = test_dir
                escape_dir = best_dir
            
            desired_vel = escape_dir * self.config.target_speed
            self.vel = 0.7 * self.vel + 0.3 * desired_vel
        else:
            self.vel += np.random.randn(2) * 0.2

        speed = np.linalg.norm(self.vel)
        if speed > self.config.target_speed:
            self.vel = self.vel / speed * self.config.target_speed

        new_pos = self.pos + self.vel * self.config.dt
        new_pos = np.clip(new_pos, [1, 1], [self.config.grid_width - 2, self.config.grid_height - 2])
        if not grid.is_obstacle(new_pos):
            self.pos = new_pos

        grid.deposit_pheromone(self.pos)
        self.update_trail()

    def get_priority(self) -> float:
        speed_score = np.linalg.norm(self.vel) / self.config.target_speed
        capture_score = 1.0 - min(self.capture_counter / self.config.capture_success_frames, 1.0)
        priority = (
            self.config.target_priority_speed_weight * speed_score
            + self.config.target_priority_capture_weight * capture_score
        )
        return priority

    def update_trail(self):
        self.trail = np.roll(self.trail, -1, axis=0)
        self.trail[-1] = self.pos

# ---------------------------- 简单神经网络（保持不变）----------------------------
class SimpleNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

    def forward(self, x):
        h = np.tanh(np.dot(x, self.W1) + self.b1)
        y = np.dot(h, self.W2) + self.b2
        return y

    def set_weights_for_capture(self):
        self.W1 = np.eye(4, 8) * 0.8
        self.b1 = np.zeros(8)
        self.W2 = np.zeros((8, 2))
        self.W2[2, 0] = 0.8
        self.W2[3, 1] = 0.8
        self.W2[2, 1] = 0.5
        self.W2[3, 0] = -0.5
        self.b2 = np.zeros(2)

# ---------------------------- 仿真控制器（集成 API 调用）----------------------------
class Simulation:
    def __init__(self, config: Config):
        self.config = config
        self.grid = CAGrid(config)
        self.step_count = 0
        self.capture_success_counter = 0
        self.captured = False

        # 初始化无人机
        self.uavs = []
        for i in range(config.num_uavs):
            attempts = 0
            while attempts < 100:
                x = np.random.uniform(10, config.grid_width - 10)
                y = np.random.uniform(10, config.grid_height - 10)
                if not self.grid.is_obstacle((x, y)):
                    break
                attempts += 1
            self.uavs.append(UAV(i, (x, y), config))

        # 初始化目标
        self.targets = []
        for i, init_pos in enumerate(config.target_initial_pos[:config.num_targets]):
            self.targets.append(Target(i, init_pos, config))

        # 神经网络
        self.nn_model = SimpleNN(config.nn_input_dim, config.nn_hidden_dim, config.nn_output_dim)
        self.nn_model.set_weights_for_capture()

        # PSO全局最优
        self.global_best_pos = self.uavs[0].pos.copy()
        self.global_best_fitness = -np.inf

        # ---------- 新增 API 相关属性 ----------
        self.api_last_call = -self.config.api_call_interval  # 上次调用步数
        self.api_actions = {}  # 缓存API返回的速度指令 {uav_id: [vx, vy]}

    # ---------- 新增：调用 DeepSeek API 获取速度指令 ----------
    def call_deepseek_api(self) -> Optional[Dict[int, np.ndarray]]:
        """
        构造当前状态提示，调用 DeepSeek API，解析返回的 JSON 数据，
        返回一个字典，键为无人机 ID，值为速度向量 (vx, vy)。
        若调用失败或格式错误，返回 None。
        """
        # 构造状态描述（简化，避免 token 过多）
        state = {
            "grid_size": [self.config.grid_width, self.config.grid_height],
            "uavs": [
                {
                    "id": u.id,
                    "pos": u.pos.tolist(),
                    "vel": u.vel.tolist(),
                    "mode": u.mode
                } for u in self.uavs
            ],
            "targets": [
                {
                    "id": t.id,
                    "pos": t.pos.tolist(),
                    "vel": t.vel.tolist()
                } for t in self.targets
            ],
            # 障碍物太多，只传递一个密度值作为示意（可根据需要扩展）
            "obstacle_density": self.grid.get_local_density(self.uavs[0].pos, 30)  # 示例
        }

        # 系统提示词
        system_prompt = (
            "You are an AI assistant controlling a swarm of UAVs to search for moving targets. "
            "The UAVs have a maximum speed of 3.0. "
            "Based on the current state, output the next velocity vector (vx, vy) for each UAV. "
            "Return a JSON object with keys 'actions', where each key is UAV id and value is [vx, vy]. "
            "Ensure that the speed does not exceed 3.0."
        )

        # 用户消息
        user_message = json.dumps(state)

        # 构造请求体
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.7,
            "max_tokens": 500,
            "response_format": {"type": "json_object"}  # 要求返回 JSON
        }

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(
                self.config.api_url,
                headers=headers,
                json=payload,
                timeout=self.config.api_timeout
            )
            response.raise_for_status()
            data = response.json()
            # 解析返回内容
            content = data["choices"][0]["message"]["content"]
            result = json.loads(content)
            actions = result.get("actions", {})
            # 转换为整数键和 numpy 数组
            action_dict = {}
            for k, v in actions.items():
                try:
                    uid = int(k)
                    arr = np.array(v, dtype=float)
                    # 限制速度
                    speed = np.linalg.norm(arr)
                    if speed > self.config.uav_speed:
                        arr = arr / speed * self.config.uav_speed
                    action_dict[uid] = arr
                except (ValueError, TypeError):
                    continue
            return action_dict
        except Exception as e:
            print(f"API call failed: {e}")
            return None

    def step(self):
        """单步仿真（集成 API 决策）"""
        if self.captured:
            return

        self.step_count += 1

        # 1. 环境更新
        self.grid.step()

        # 2. 目标移动
        for t in self.targets:
            t.step(self.uavs, self.grid)

        # 3. 检测目标是否可见
        any_visible = False
        for u in self.uavs:
            u.target_visible = False
            for t in self.targets:
                dist = np.linalg.norm(u.pos - t.pos)
                if dist < self.config.uav_sensor_range:
                    u.target_visible = True
                    any_visible = True
                    break

        # 切换模式
        for u in self.uavs:
            u.mode = 'capture' if any_visible else 'search'

        # 4. 更新 PSO 适应度（始终需要，即使使用 API 也可能降级）
        for u in self.uavs:
            fitness = u.sense_pheromone(self.grid) + np.random.rand() * 0.1
            u.update_best(fitness)
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_pos = u.pos.copy()

        # 5. 无人机运动决策（优先使用 API，否则回退到原有逻辑）
        if self.config.use_api and not any_visible:  # 仅在搜索阶段调用 API
            # 判断是否到达调用间隔
            if self.step_count - self.api_last_call >= self.config.api_call_interval:
                actions = self.call_deepseek_api()
                if actions is not None:
                    self.api_actions = actions
                    self.api_last_call = self.step_count
                else:
                    # API 失败，清空缓存，后续步将使用 PSO
                    self.api_actions = {}
            # 如果存在缓存的动作，则应用
            if self.api_actions:
                for u in self.uavs:
                    if u.id in self.api_actions:
                        u.vel = self.api_actions[u.id].copy()
                        # 应用速度后，简单避障和边界约束
                        new_pos = u.pos + u.vel * self.config.dt
                        new_pos = u._simple_avoid_obstacle(new_pos, self.grid)
                        new_pos = u._avoid_boundary(new_pos)
                        u.pos = new_pos
                        self.grid.mark_searched(u.pos)
                    else:
                        # 如果某个无人机没有指令，降级到 PSO
                        u.pso_update(self.global_best_pos, self.grid, self.uavs)
            else:
                # 没有缓存动作，使用原有 PSO
                for u in self.uavs:
                    u.pso_update(self.global_best_pos, self.grid, self.uavs)
        else:
            # 未启用 API 或处于抓捕模式，使用原有逻辑
            if any_visible:
                assignments = self.assign_targets_hungarian()
                for i, u in enumerate(self.uavs):
                    target = self.targets[assignments[i]]
                    u.neural_control(target.pos, target.vel, self.nn_model, self.grid)
            else:
                for u in self.uavs:
                    u.pso_update(self.global_best_pos, self.grid, self.uavs)

        # 6. 同伴排斥（弱化）
        for u in self.uavs:
            crowd_repel = u.avoid_crowd(self.uavs)
            u.vel += crowd_repel * 0.5
            speed = np.linalg.norm(u.vel)
            if speed > self.config.uav_speed:
                u.vel = u.vel / speed * self.config.uav_speed

        # 7. 避碰（弱化）
        for u in self.uavs:
            u.avoid_collision(self.uavs, self.grid)

        # 8. 更新轨迹
        for u in self.uavs:
            u.update_trail()

        # 9. 检测抓捕成功（与原有相同）
        if any_visible:
            dists = []
            for u in self.uavs:
                nearest = min(self.targets, key=lambda t: np.linalg.norm(t.pos - u.pos))
                dists.append(np.linalg.norm(u.pos - nearest.pos))
            if all(d < self.config.capture_success_distance for d in dists):
                self.capture_success_counter += 1
                if self.capture_success_counter >= self.config.capture_success_frames:
                    self.captured = True
            else:
                self.capture_success_counter = 0

    def assign_targets_hungarian(self) -> List[int]:
        """匈牙利算法分配目标（与原有相同）"""
        num_uavs = len(self.uavs)
        num_targets = len(self.targets)

        target_priorities = np.array([t.get_priority() for t in self.targets])
        priority_max = max(target_priorities.max(), 1e-6)
        target_priorities = target_priorities / priority_max

        dist_matrix = np.zeros((num_uavs, num_targets))
        for i, u in enumerate(self.uavs):
            for j, t in enumerate(self.targets):
                dist = np.linalg.norm(u.pos - t.pos)
                dist_matrix[i, j] = dist / self.config.grid_width - target_priorities[j]

        if num_uavs > num_targets:
            repeat_times = math.ceil(num_uavs / num_targets)
            dist_matrix = np.tile(dist_matrix, (1, repeat_times))[:, :num_uavs]

        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        assignments = [col % len(self.targets) for col in col_ind]
        return assignments

# ---------------------------- 可视化模块（保持不变）----------------------------
class Visualizer:
    # ... 原有代码保持不变 ...
    def __init__(self, sim: Simulation):
        self.sim = sim
        self.config = sim.config
        sns.set_style("whitegrid")
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.pheromone_img = None
        self.uav_scatter = None
        self.target_scatters = []
        self.target_trails = []
        self.uav_trails = []
        self.sensor_circles = []
        self.obstacle_scatter = None
        self.title_text = None
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    def init_plot(self):
        self.ax.set_xlim(0, self.config.grid_width)
        self.ax.set_ylim(0, self.config.grid_height)
        self.ax.set_aspect('equal')
        self.ax.set_title("UAV Swarm Capture Simulation with DeepSeek API", fontsize=16)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        if self.config.show_grid:
            self.ax.grid(True, linestyle=':', alpha=0.5)

        obs_pos = self.sim.grid.get_obstacle_positions()
        if len(obs_pos) > 0:
            self.obstacle_scatter = self.ax.scatter(obs_pos[:, 0], obs_pos[:, 1],
                                                    c=self.config.obstacle_color, marker='s', s=30,
                                                    alpha=self.config.obstacle_alpha, label='Obstacles', zorder=1)

        self.pheromone_img = self.ax.imshow(
            self.sim.grid.pheromone,
            origin='lower',
            extent=[0, self.config.grid_width, 0, self.config.grid_height],
            cmap=self.config.pheromone_cmap,
            alpha=self.config.pheromone_alpha,
            vmin=0,
            zorder=2
        )
        plt.colorbar(self.pheromone_img, ax=self.ax, label='Pheromone', fraction=0.046, pad=0.04)

        for i, u in enumerate(self.sim.uavs):
            line, = self.ax.plot([], [], color=self.config.uav_search_color,
                                 alpha=self.config.trail_alpha, linewidth=1)
            self.uav_trails.append(line)

        uav_positions = np.array([u.pos for u in self.sim.uavs])
        self.uav_scatter = self.ax.scatter(uav_positions[:, 0], uav_positions[:, 1],
                                           c=self.config.uav_search_color, s=80,
                                           edgecolors=self.config.uav_edgecolor, linewidth=1.5,
                                           label='UAVs', zorder=5)

        if self.config.show_sensor_range:
            for i, u in enumerate(self.sim.uavs):
                circle = Circle((u.pos[0], u.pos[1]), self.config.uav_sensor_range,
                                color='cyan', fill=False, linestyle='--', linewidth=1, alpha=0.5)
                self.ax.add_patch(circle)
                self.sensor_circles.append(circle)

        colors = self.config.target_colors
        for i, t in enumerate(self.sim.targets):
            col = colors[i % len(colors)]
            line, = self.ax.plot([], [], color=col, alpha=self.config.trail_alpha,
                                 linewidth=1.5, linestyle=':')
            self.target_trails.append(line)
            scatter = self.ax.scatter(t.pos[0], t.pos[1], c=col, s=150,
                                      marker=self.config.target_marker,
                                      edgecolors=self.config.target_edgecolor, linewidth=2,
                                      label=f'Target {i+1}' if i==0 else "", zorder=10)
            self.target_scatters.append(scatter)

        self.title_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                        fontsize=12, verticalalignment='top',
                                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='UAV',
                   markerfacecolor=self.config.uav_search_color, markeredgecolor=self.config.uav_edgecolor, markersize=10),
            Line2D([0], [0], marker=self.config.target_marker, color='w', label='Target',
                   markerfacecolor=self.config.target_colors[0], markeredgecolor=self.config.target_edgecolor, markersize=12),
            Patch(facecolor=self.config.obstacle_color, alpha=self.config.obstacle_alpha, label='Obstacle')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right')

    def update(self, frame):
        for _ in range(self.config.plot_interval):
            self.sim.step()

        self.pheromone_img.set_data(self.sim.grid.pheromone)

        uav_positions = np.array([u.pos for u in self.sim.uavs])
        self.uav_scatter.set_offsets(uav_positions)

        if self.config.show_trails:
            for i, u in enumerate(self.sim.uavs):
                self.uav_trails[i].set_data(u.trail[:, 0], u.trail[:, 1])

        if self.config.show_sensor_range:
            for i, u in enumerate(self.sim.uavs):
                self.sensor_circles[i].center = (u.pos[0], u.pos[1])

        for i, t in enumerate(self.sim.targets):
            self.target_scatters[i].set_offsets(t.pos)
            if self.config.show_trails:
                self.target_trails[i].set_data(t.trail[:, 0], t.trail[:, 1])

        colors = [self.config.uav_capture_color if u.mode == 'capture' else self.config.uav_search_color for u in self.sim.uavs]
        self.uav_scatter.set_color(colors)

        visible_count = sum(
            any(np.linalg.norm(u.pos - t.pos) < self.config.uav_sensor_range for u in self.sim.uavs)
            for t in self.sim.targets
        )
        if self.sim.targets:
            highest_priority_target = max(self.sim.targets, key=lambda t: t.get_priority())
            priority_info = f" | Priority Target: {highest_priority_target.id+1}"
        else:
            priority_info = ""
        
        mode_str = 'Capture' if any(u.mode == 'capture' for u in self.sim.uavs) else 'Search'
        status = f"Step: {self.sim.step_count} | Mode: {mode_str} | Visible Targets: {visible_count}/{len(self.sim.targets)}{priority_info}"
        if self.sim.captured:
            status = "ALL TARGETS CAPTURED! " + status
        self.title_text.set_text(status)

        artists = [self.pheromone_img, self.uav_scatter] + self.target_scatters + self.uav_trails + self.target_trails + self.sensor_circles
        if self.obstacle_scatter:
            artists.append(self.obstacle_scatter)
        return artists

    def animate(self, show=True):
        self.init_plot()
        anim = FuncAnimation(self.fig, self.update, frames=range(500), interval=30, blit=False)
        if show:
            plt.show()
        return anim

# ---------------------------- 主程序 ----------------------------
if __name__ == "__main__":
    config = Config()
    config.target_initial_pos = [(20,20), (80,80), (50,70), (30,80)]
    config.num_targets = 4
    # 启用 API 决策（请先设置有效的 API Key）
    config.use_api = True
    config.api_key = "****"  # 替换为真实密钥
    config.api_call_interval = 10              # 每 10 步调用一次

    sim = Simulation(config)
    viz = Visualizer(sim)

    # 生成动画
    anim = viz.animate(show=True)
    # 如需保存GIF，取消注释
    anim.save('uav_capture_api.gif', writer='pillow', fps=20, dpi=100)