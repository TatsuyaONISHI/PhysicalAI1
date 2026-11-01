# PAI最終課題: LiDAR + 模倣学習AIによる差動二輪ロボットの自律ナビゲーション

PyBulletでシミュレーションしたスキッドステア(差動二輪と同じ指令で動く)ロボットが、
LiDAR風の距離センサとニューラルネットの推論だけで障害物を避けながら複数のゴールを巡回します。

## ベースラインからの発展
ベースライン「ROS2とGazeboの連携によるロボット操作」(Qiita, N622) では `/cmd_vel` の
`linear.x` / `angular.z` を人間のキー入力で与えていました。本作では同じ (v, ω) 指令を

1. 36本ビームのLiDAR風センサ + ゴール方向を入力とし、
2. 模倣学習 (Behavior Cloning) + DAgger で学習したMLPが推論して出力します。

## 実行方法
- Google Colab: `PAI_final_ai_nav.ipynb` を開いて上から順に実行
- ローカル (Python 3.11推奨): `pip install -r requirements.txt` → `python run_all.py`

## 結果 (20環境 × 3ゴール = 60試行)
| 方式 | ゴール到達率 | 接触ありの試行 |
|---|---|---|
| お手本コントローラ (ギャップ探索 + 斥力) | 96.7% | 5.0% |
| Behavior Cloning のみ | 73.3% | 15.0% |
| + DAgger 1回 | 83.3% | 20.0% |
| + DAgger 2回 (最終) | 88.3% | 11.7% |

デモ動画 (seed=10, 等速): 9ゴール中9到達, 約101秒。

## ファイル
- `ai_nav.py` 環境・センサ・お手本・学習・動画生成
- `PAI_final_ai_nav.ipynb` Colab用ノートブック
- `run_all.py` 一括実行スクリプト
