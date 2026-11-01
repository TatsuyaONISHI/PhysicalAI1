# -*- coding: utf-8 -*-
"""
PAI最終課題: LiDAR風センサ + 模倣学習(Behavior Cloning)ニューラルネットによる
差動二輪(スキッドステア)ロボットの自律ナビゲーション (PyBullet / Google Colab対応)

ベースライン: 「ROS2とGazeboの連携によるロボット操作」(/cmd_vel の linear.x, angular.z で
キー操作する差動二輪ロボット)。本コードではその速度指令(v, ω)を人間のキー入力ではなく
AI(ニューラルネット)の推論で生成し、障害物を避けながら複数のゴールを巡回させる。
"""
import math
import time
import numpy as np
import pybullet as p
import pybullet_data
import cv2
import imageio

# ----------------------------------------------------------------------------
# パラメータ
# ----------------------------------------------------------------------------
ARENA = 5.0            # アリーナ半径(正方形の半辺) [m]
N_RAYS = 36            # LiDARのビーム本数(10度刻み)
RAY_RANGE = 4.0        # LiDARの最大距離 [m]
RAY_START = 0.55       # 車体自身に当たらないようにする開始半径 [m]
CTRL_HZ = 10           # 制御周期 [Hz]
SIM_HZ = 240
SUBSTEPS = SIM_HZ // CTRL_HZ
V_MAX = 1.0            # 最大並進速度 [m/s]  (= linear.x)
W_MAX = 1.5            # 最大角速度 [rad/s]  (= angular.z)
GOAL_TOL = 0.5         # ゴール到達判定 [m]
WHEEL_R = 0.165        # Husky車輪半径 [m]
TRACK = 0.555          # 左右車輪間隔 [m]
SKID = 1.8             # スキッドステア旋回の補正係数


# ----------------------------------------------------------------------------
# 環境
# ----------------------------------------------------------------------------
class NavEnv:
    def __init__(self, seed=0, n_obstacles=8, gui=False):
        self.rng = np.random.default_rng(seed)
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.n_obstacles = n_obstacles
        self.robot = None
        self.reset()

    # --- ワールド構築 -------------------------------------------------------
    def reset(self, obstacles=None, start=None):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1.0 / SIM_HZ)
        p.loadURDF("plane.urdf")
        self._make_walls()
        self.obstacles = []  # (x, y, half_x, half_y)
        if obstacles is None:
            obstacles = self._random_obstacles()
        for (x, y, hx, hy) in obstacles:
            self._add_box(x, y, hx, hy, color=[0.35, 0.45, 0.85, 1])
            self.obstacles.append((x, y, hx, hy))
        if start is None:
            start = self.sample_free_point()
        yaw = self.rng.uniform(-math.pi, math.pi)
        self.robot = p.loadURDF("husky/husky.urdf", [start[0], start[1], 0.15],
                                p.getQuaternionFromEuler([0, 0, yaw]))
        self._find_wheels()
        self.goal = None
        self.goal_marker = None
        self.path = []
        for _ in range(50):  # 着地させる
            p.stepSimulation()

    def _make_walls(self):
        t, h = 0.1, 0.4
        L = ARENA
        for (x, y, hx, hy) in [(0, L, L, t), (0, -L, L, t), (L, 0, t, L), (-L, 0, t, L)]:
            self._add_box(x, y, hx, hy, h=h, color=[0.6, 0.6, 0.6, 1])

    def _add_box(self, x, y, hx, hy, h=0.4, color=(0.5, 0.5, 0.8, 1)):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, h])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, h], rgbaColor=color)
        return p.createMultiBody(0, col, vis, [x, y, h])

    def _random_obstacles(self):
        obs = []
        tries = 0
        while len(obs) < self.n_obstacles and tries < 500:
            tries += 1
            hx, hy = self.rng.uniform(0.2, 0.6, size=2)
            x, y = self.rng.uniform(-ARENA + 1.0, ARENA - 1.0, size=2)
            ok = all(abs(x - ox) > hx + ohx + 0.9 or abs(y - oy) > hy + ohy + 0.9
                     for (ox, oy, ohx, ohy) in obs)
            if ok:
                obs.append((x, y, hx, hy))
        return obs

    def is_free(self, x, y, margin=0.8):
        if abs(x) > ARENA - margin or abs(y) > ARENA - margin:
            return False
        for (ox, oy, hx, hy) in self.obstacles:
            if abs(x - ox) < hx + margin and abs(y - oy) < hy + margin:
                return False
        return True

    def sample_free_point(self, min_dist_from=None, min_dist=0.0):
        for _ in range(1000):
            x, y = self.rng.uniform(-ARENA + 0.8, ARENA - 0.8, size=2)
            if not self.is_free(x, y):
                continue
            if min_dist_from is not None and math.hypot(x - min_dist_from[0], y - min_dist_from[1]) < min_dist:
                continue
            return (x, y)
        return (0.0, 0.0)

    def set_goal(self, g):
        self.goal = g
        if self.goal_marker is not None:
            p.removeBody(self.goal_marker)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.3, length=0.02, rgbaColor=[1, 0.2, 0.2, 0.9])
        self.goal_marker = p.createMultiBody(0, -1, vis, [g[0], g[1], 0.01])

    def _find_wheels(self):
        self.left, self.right = [], []
        for j in range(p.getNumJoints(self.robot)):
            name = p.getJointInfo(self.robot, j)[1].decode()
            if "wheel" in name:
                (self.left if "left" in name else self.right).append(j)

    # --- センサ ------------------------------------------------------------
    def pose(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        yaw = p.getEulerFromQuaternion(orn)[2]
        return pos[0], pos[1], yaw

    def lidar(self):
        x, y, yaw = self.pose()
        z = 0.3
        angles = yaw + np.linspace(-math.pi, math.pi, N_RAYS, endpoint=False)
        starts = [[x + RAY_START * math.cos(a), y + RAY_START * math.sin(a), z] for a in angles]
        ends = [[x + RAY_RANGE * math.cos(a), y + RAY_RANGE * math.sin(a), z] for a in angles]
        res = p.rayTestBatch(starts, ends)
        d = np.array([RAY_START + r[2] * (RAY_RANGE - RAY_START) if r[0] != -1 else RAY_RANGE
                      for r in res])
        return d

    def observation(self):
        """AIへの入力: 正規化LiDAR距離(36) + ゴールまでの距離 + ゴール方向のsin/cos"""
        x, y, yaw = self.pose()
        d = self.lidar()
        gx, gy = self.goal
        dist = math.hypot(gx - x, gy - y)
        ang = math.atan2(gy - y, gx - x) - yaw
        obs = np.concatenate([d / RAY_RANGE,
                              [min(dist, 6.0) / 6.0, math.sin(ang), math.cos(ang)]])
        return obs.astype(np.float32), d, dist, ang

    # --- アクチュエータ(= /cmd_vel の linear.x, angular.z) ---------------
    def apply_cmd(self, v, w):
        v = float(np.clip(v, -V_MAX, V_MAX))
        w = float(np.clip(w, -W_MAX, W_MAX))
        vl = (v - w * TRACK * SKID / 2) / WHEEL_R
        vr = (v + w * TRACK * SKID / 2) / WHEEL_R
        for j in self.left:
            p.setJointMotorControl2(self.robot, j, p.VELOCITY_CONTROL, targetVelocity=vl, force=60)
        for j in self.right:
            p.setJointMotorControl2(self.robot, j, p.VELOCITY_CONTROL, targetVelocity=vr, force=60)
        for _ in range(SUBSTEPS):
            p.stepSimulation()
        x, y, _ = self.pose()
        self.path.append((x, y))

    def min_clearance(self):
        x, y, _ = self.pose()
        best = 99.0
        for (ox, oy, hx, hy) in self.obstacles + [(0, ARENA, ARENA, 0.1), (0, -ARENA, ARENA, 0.1),
                                                  (ARENA, 0, 0.1, ARENA), (-ARENA, 0, 0.1, ARENA)]:
            dx = max(abs(x - ox) - hx, 0)
            dy = max(abs(y - oy) - hy, 0)
            best = min(best, math.hypot(dx, dy) - 0.45)  # 0.45: 車体の外接半径の目安
        return best


# ----------------------------------------------------------------------------
# エキスパート(教師): ギャップ探索 + ポテンシャル場の反応型コントローラ
# ----------------------------------------------------------------------------
RAY_ANGLES = np.linspace(-math.pi, math.pi, N_RAYS, endpoint=False)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def expert_policy(d, dist, ang, need=2.5, wid=1, kw=2.2, infl=1.2, kr=0.5):
    """ギャップ探索(VFH風) + 近接斥力 によるお手本コントローラ。
    d: LiDAR距離(36), dist: ゴール距離, ang: ゴール方向(ロボット座標系)"""
    n = N_RAYS
    # 各方向について、左右wid本を含む窓の最小距離 = 車体が通れる余裕
    win = np.array([min(d[(i + k) % n] for k in range(-wid, wid + 1)) for i in range(n)])
    free = win >= min(dist, need)
    if free.any():
        diffs = np.abs([wrap(a - ang) for a in RAY_ANGLES])
        diffs[~free] = 9.0
        tgt = RAY_ANGLES[int(np.argmin(diffs))]   # ゴールに最も近い「通れる方向」
        if abs(wrap(tgt - ang)) < math.radians(10):
            tgt = ang
    else:
        tgt = RAY_ANGLES[int(np.argmax(win))]     # 全部塞がっていたら一番開けた方向
    vec = np.array([math.cos(tgt), math.sin(tgt)])
    for a, r in zip(RAY_ANGLES, d):               # 近い障害物から離れる斥力
        if r < infl:
            vec -= (1.0 / max(r, 0.3) - 1.0 / infl) ** 2 * kr * np.array([math.cos(a), math.sin(a)])
    heading = math.atan2(vec[1], vec[0])
    w = float(np.clip(kw * heading, -W_MAX, W_MAX))
    front = np.min(d[n // 2 - 3: n // 2 + 4])     # 正面±30度
    v = V_MAX * max(math.cos(heading), 0.0) ** 2
    v *= np.clip((front - 0.55) / 1.0, 0.0, 1.0)
    v *= np.clip(dist, 0.35, 1.0)
    if abs(heading) > 1.0:                        # 大きく向きが違うときはその場旋回
        v = 0.0
    return np.array([v, w], dtype=np.float32)


def expert_from_obs(obs):
    d = obs[:N_RAYS] * RAY_RANGE
    dist = obs[N_RAYS] * 6.0
    ang = math.atan2(obs[N_RAYS + 1], obs[N_RAYS + 2])
    return expert_policy(d, dist, ang)


# ----------------------------------------------------------------------------
# データ収集 / 学習
# ----------------------------------------------------------------------------
def collect(env_seeds, steps_per_goal=200, goals_per_env=4, policy=None, beta=1.0, rng=None):
    """エキスパートで走らせて (観測, 行動) を集める。
    policy を与えると DAgger: 学習済みモデルで走らせつつ、ラベルはエキスパートが付ける"""
    rng = rng or np.random.default_rng(0)
    X, Y = [], []
    for s in env_seeds:
        env = NavEnv(seed=s)
        for _ in range(goals_per_env):
            x, y, _ = env.pose()
            env.set_goal(env.sample_free_point((x, y), 3.0))
            for _ in range(steps_per_goal):
                obs, d, dist, ang = env.observation()
                if dist < GOAL_TOL:
                    break
                a_exp = expert_policy(d, dist, ang)
                X.append(obs); Y.append(a_exp)
                if policy is not None and rng.random() > beta:
                    a = policy(obs)
                else:
                    a = a_exp
                env.apply_cmd(*a)
                if env.min_clearance() < 0.0:
                    break
        p.disconnect()
    return np.array(X), np.array(Y)


def train(X, Y):
    from sklearn.neural_network import MLPRegressor
    model = MLPRegressor(hidden_layer_sizes=(128, 128), activation="relu",
                         learning_rate_init=1e-3, max_iter=300, early_stopping=True,
                         random_state=0)
    Ys = Y / np.array([V_MAX, W_MAX])
    model.fit(X, Ys)
    return model


def make_policy(model):
    def policy(obs):
        a = model.predict(obs[None])[0] * np.array([V_MAX, W_MAX])
        return a
    return policy


# ----------------------------------------------------------------------------
# 可視化(動画)
# ----------------------------------------------------------------------------
W, H = 640, 400
MAP = 400


def render_camera(env):
    x, y, yaw = env.pose()
    eye = [x - 2.5 * math.cos(yaw), y - 2.5 * math.sin(yaw), 3.2]
    tgt = [x + 1.5 * math.cos(yaw), y + 1.5 * math.sin(yaw), 0.0]
    view = p.computeViewMatrix(eye, tgt, [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(70, W / H, 0.1, 30)
    img = p.getCameraImage(W, H, view, proj, renderer=p.ER_TINY_RENDERER)[2]
    img = np.reshape(img, (H, W, 4))[:, :, :3].astype(np.uint8)
    return np.ascontiguousarray(img)


def to_px(x, y):
    s = MAP / (2 * ARENA + 0.4)
    return int((x + ARENA + 0.2) * s), int((ARENA + 0.2 - y) * s)


def render_map(env, d, a, goals_done):
    m = np.full((MAP, MAP, 3), 245, np.uint8)
    for (ox, oy, hx, hy) in env.obstacles:
        cv2.rectangle(m, to_px(ox - hx, oy + hy), to_px(ox + hx, oy - hy), (90, 115, 215), -1)
    cv2.rectangle(m, to_px(-ARENA, ARENA), to_px(ARENA, -ARENA), (120, 120, 120), 2)
    x, y, yaw = env.pose()
    # LiDAR
    for ang, r in zip(np.linspace(-math.pi, math.pi, N_RAYS, endpoint=False), d):
        ex, ey = x + r * math.cos(yaw + ang), y + r * math.sin(yaw + ang)
        col = (230, 60, 60) if r < 1.0 else (90, 200, 90)
        cv2.line(m, to_px(x, y), to_px(ex, ey), col, 1)
    # 軌跡
    pts = [to_px(px, py) for px, py in env.path[-2000:]]
    for i in range(1, len(pts)):
        cv2.line(m, pts[i - 1], pts[i], (40, 40, 40), 2)
    # ゴール
    gx, gy = env.goal
    cv2.circle(m, to_px(gx, gy), 9, (230, 40, 40), -1)
    # ロボット
    cv2.circle(m, to_px(x, y), 8, (30, 150, 30), -1)
    cv2.line(m, to_px(x, y), to_px(x + 0.6 * math.cos(yaw), y + 0.6 * math.sin(yaw)), (0, 0, 0), 2)
    return m


def overlay(frame, lines):
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (10, 25 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, t, (10, 25 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def run_demo(policy, seed, out_path, title, target_sec=90, max_sec=170,
             max_steps_per_goal=300, fps=CTRL_HZ):
    """AIの推論だけでゴールを次々に巡回し、その様子をMP4に保存する。
    動画が target_sec 秒を超えたら、そのゴールを最後に終了(課題要件: 1〜3分)"""
    env = NavEnv(seed=seed)
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=7, macro_block_size=8)
    reached, goals, frames = 0, 0, 0
    while frames < target_sec * fps:
        goals += 1
        x, y, _ = env.pose()
        env.set_goal(env.sample_free_point((x, y), 4.0))
        for k in range(max_steps_per_goal):
            obs, d, dist, ang = env.observation()
            if dist < GOAL_TOL:
                reached += 1
                break
            a = policy(obs)
            env.apply_cmd(*a)
            cam = render_camera(env)
            mp = render_map(env, d, a, reached)
            frame = np.concatenate([cam, cv2.resize(mp, (H, H))], axis=1)
            overlay(frame, [title,
                            f"goal #{goals}   reached {reached}",
                            f"cmd_vel  v={a[0]:+.2f} m/s  w={a[1]:+.2f} rad/s",
                            f"goal dist {dist:.2f} m"])
            writer.append_data(frame)
            frames += 1
            if frames >= max_sec * fps:
                break
        if frames >= max_sec * fps:
            break
    writer.close()
    p.disconnect()
    return reached, goals, frames / fps


def evaluate(policy, seeds, n_goals=3, max_steps=300):
    ok, tot, col = 0, 0, 0
    for s in seeds:
        env = NavEnv(seed=s)
        for _ in range(n_goals):
            x, y, _ = env.pose()
            env.set_goal(env.sample_free_point((x, y), 4.0))
            tot += 1
            hit = False
            for _ in range(max_steps):
                obs, d, dist, ang = env.observation()
                if dist < GOAL_TOL:
                    ok += 1
                    break
                env.apply_cmd(*policy(obs))
                if env.min_clearance() < 0.0:
                    hit = True
            col += hit
        p.disconnect()
    return ok / tot, col / tot
