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

## 結果
Google Colab (Python 3.12, CPU) で `PAI_final_ai_nav.ipynb` を実行した結果です。
評価は学習に使っていない20環境 × 3ゴール = 60試行。

| 方式 | ゴール到達率 | 接触ありの試行 | 学習データ数 |
|---|---|---|---|
| お手本コントローラ (ギャップ探索 + 斥力) | 96.7% | 5.0% | - |
| Behavior Cloning のみ | 71.7% | 11.7% | 13,408 |
| + DAgger 1回 | 86.7% | 21.7% | 24,506 |
| + DAgger 2回 (最終) | 90.0% | 21.7% | 35,615 |

デモ動画 (seed=10, 等速再生): 9ゴール中8ゴール到達, 約107秒。

実行時間の目安 (Colab): インストール 約10分 / BC 約3分 / DAgger 約9分 / 動画生成 約6分。

※ PyBullet や scikit-learn のバージョンによって数値は多少変わることがあります。

## ファイル
- `ai_nav.py` 環境・センサ・お手本・学習・動画生成
- `PAI_final_ai_nav.ipynb` Colab用ノートブック
- `run_all.py` 一括実行スクリプト
