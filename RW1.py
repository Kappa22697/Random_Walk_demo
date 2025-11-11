#!/usr/bin/env python3
"""
ランダムウォークのシミュレーション（確率パラメータ設定可能版）
Random Walk Simulation with Configurable Probabilities
"""

import numpy as np
import matplotlib.pyplot as plt
import sys


# ========================================
# パラメータ設定（ここを編集してください）
# ========================================

# シミュレーションの基本設定
SIMULATION_CONFIG = {
    'dimension': 2,        # 次元: 1 または 2
    'n_steps': 1000,       # ステップ数
    'n_walkers': 1,        # ウォーカーの数（1なら単独、複数なら集団シミュレーション）
}

# 1次元ランダムウォークの確率設定
# （合計が1.0になるように設定してください）
CONFIG_1D = {
    'p_left': 0.3,         # 左に移動する確率
    'p_stay': 0.4,         # 停滞する確率
    'p_right': 0.3,        # 右に移動する確率
}

# 2次元ランダムウォークの設定
CONFIG_2D = {
    'mode': 'lattice',     # 'continuous' (連続角度) または 'lattice' (格子型)
    'p_stay': 0.1,         # 停滞する確率（continuousモードの場合）

    # latticeモードの場合の各方向の確率（合計が1.0になるように設定）
    'p_right': 0.35,       # 右に移動する確率
    'p_up': 0.35,          # 上に移動する確率
    'p_left': 0.1,         # 左に移動する確率
    'p_down': 0.1,         # 下に移動する確率
}

# ========================================
# 以下、プログラム本体
# ========================================


class RandomWalk1D:
    """1次元ランダムウォーク（停滞確率付き）"""

    def __init__(self, n_steps=1000, step_size=1, p_left=0.33, p_right=0.33, p_stay=0.34):
        """
        Args:
            n_steps: ステップ数
            step_size: 1ステップあたりの移動距離
            p_left: 左に移動する確率
            p_right: 右に移動する確率
            p_stay: 停滞する確率
        """
        self.n_steps = n_steps
        self.step_size = step_size
        self.p_left = p_left
        self.p_right = p_right
        self.p_stay = p_stay
        self.positions = None

        # 確率の合計チェック
        total = p_left + p_right + p_stay
        if not np.isclose(total, 1.0):
            raise ValueError(f"確率の合計が1になりません: {total}")

    def simulate(self):
        """ランダムウォークをシミュレート"""
        # -1(左), 0(停滞), +1(右) のランダムな選択
        choices = [-1, 0, 1]
        probabilities = [self.p_left, self.p_stay, self.p_right]

        steps = np.random.choice(choices, size=self.n_steps, p=probabilities) * self.step_size

        # 累積和で位置を計算
        self.positions = np.concatenate([[0], np.cumsum(steps)])
        return self.positions

    def plot(self):
        """結果をプロット"""
        if self.positions is None:
            self.simulate()

        plt.figure(figsize=(12, 6))
        plt.plot(self.positions, linewidth=0.8, alpha=0.7)
        plt.xlabel('Step', fontsize=12)
        plt.ylabel('Position', fontsize=12)
        plt.title(f'1D Random Walk (Left={self.p_left:.2f}, Stay={self.p_stay:.2f}, Right={self.p_right:.2f})',
                 fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=0.5, alpha=0.5)

        # 統計情報を表示
        final_pos = self.positions[-1]
        max_pos = np.max(self.positions)
        min_pos = np.min(self.positions)

        stats_text = f'Final Position: {final_pos:.2f}\nMax: {max_pos:.2f}\nMin: {min_pos:.2f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()


class RandomWalk2D:
    """2次元ランダムウォーク（停滞確率付き）"""

    def __init__(self, n_steps=1000, step_size=1, p_stay=0.0, mode='continuous',
                 p_right=None, p_up=None, p_left=None, p_down=None):
        """
        Args:
            n_steps: ステップ数
            step_size: 1ステップあたりの移動距離
            p_stay: 停滞する確率
            mode: 'continuous' (連続角度) または 'lattice' (4方向格子)
            p_right: 右に移動する確率（latticeモードのみ）
            p_up: 上に移動する確率（latticeモードのみ）
            p_left: 左に移動する確率（latticeモードのみ）
            p_down: 下に移動する確率（latticeモードのみ）
        """
        self.n_steps = n_steps
        self.step_size = step_size
        self.p_stay = p_stay
        self.mode = mode
        self.x_positions = None
        self.y_positions = None

        # latticeモードの確率設定
        if mode == 'lattice' and all(p is not None for p in [p_right, p_up, p_left, p_down]):
            self.p_right = p_right
            self.p_up = p_up
            self.p_left = p_left
            self.p_down = p_down

            # 確率の合計チェック
            total = p_stay + p_right + p_up + p_left + p_down
            if not np.isclose(total, 1.0):
                raise ValueError(f"確率の合計が1になりません: {total}")
        else:
            # デフォルト：等確率
            if mode == 'lattice':
                p_move = (1 - p_stay) / 4
                self.p_right = p_move
                self.p_up = p_move
                self.p_left = p_move
                self.p_down = p_move
            else:
                self.p_right = None
                self.p_up = None
                self.p_left = None
                self.p_down = None

        if not (0 <= p_stay <= 1):
            raise ValueError(f"停滞確率は0以上1以下である必要があります: {p_stay}")

    def simulate(self):
        """ランダムウォークをシミュレート"""
        if self.mode == 'continuous':
            # 連続角度モード
            x_steps = []
            y_steps = []

            for _ in range(self.n_steps):
                # 停滞するかどうかを判定
                if np.random.random() < self.p_stay:
                    x_steps.append(0)
                    y_steps.append(0)
                else:
                    # 連続角度（全方向）
                    angle = np.random.uniform(0, 2*np.pi)
                    x_steps.append(np.cos(angle) * self.step_size)
                    y_steps.append(np.sin(angle) * self.step_size)

        elif self.mode == 'lattice':
            # 格子型モード（確率設定を使用）
            choices = [0, 1, 2, 3, 4]  # 0:右, 1:上, 2:左, 3:下, 4:停滞
            probabilities = [self.p_right, self.p_up, self.p_left, self.p_down, self.p_stay]

            steps = np.random.choice(choices, size=self.n_steps, p=probabilities)

            x_steps = []
            y_steps = []
            for step in steps:
                if step == 0:  # 右
                    x_steps.append(self.step_size)
                    y_steps.append(0)
                elif step == 1:  # 上
                    x_steps.append(0)
                    y_steps.append(self.step_size)
                elif step == 2:  # 左
                    x_steps.append(-self.step_size)
                    y_steps.append(0)
                elif step == 3:  # 下
                    x_steps.append(0)
                    y_steps.append(-self.step_size)
                else:  # 停滞
                    x_steps.append(0)
                    y_steps.append(0)

        # 累積和で位置を計算
        self.x_positions = np.concatenate([[0], np.cumsum(x_steps)])
        self.y_positions = np.concatenate([[0], np.cumsum(y_steps)])

        return self.x_positions, self.y_positions

    def plot(self):
        """結果をプロット"""
        if self.x_positions is None:
            self.simulate()

        plt.figure(figsize=(10, 10))

        colors = np.arange(len(self.x_positions))

        plt.plot(self.x_positions, self.y_positions, linewidth=0.5, alpha=0.3, color='gray')
        plt.scatter(self.x_positions, self.y_positions, c=colors, cmap='viridis',
                   s=1, alpha=0.5)

        # スタートとゴールをマーク
        plt.scatter(0, 0, c='green', s=200, marker='o',
                   edgecolors='black', linewidths=2, label='Start', zorder=5)
        plt.scatter(self.x_positions[-1], self.y_positions[-1], c='red', s=200,
                   marker='x', linewidths=3, label='End', zorder=5)

        plt.xlabel('X Position', fontsize=12)
        plt.ylabel('Y Position', fontsize=12)

        if self.mode == 'continuous':
            mode_text = f'Continuous, Stay={self.p_stay:.2f}'
        else:
            mode_text = f'Lattice: R={self.p_right:.2f}, U={self.p_up:.2f}, L={self.p_left:.2f}, D={self.p_down:.2f}, S={self.p_stay:.2f}'
        plt.title(f'2D Random Walk\n{mode_text}', fontsize=14)

        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.legend(fontsize=10)
        plt.colorbar(plt.scatter([], [], c=[], cmap='viridis'),
                    label='Steps', shrink=0.8)

        # 統計情報
        final_dist = np.sqrt(self.x_positions[-1]**2 + self.y_positions[-1]**2)
        stats_text = f'Final Distance: {final_dist:.2f}\nFinal Position: ({self.x_positions[-1]:.2f}, {self.y_positions[-1]:.2f})'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()

    def get_distance_from_origin(self):
        """原点からの距離を計算"""
        if self.x_positions is None:
            self.simulate()
        return np.sqrt(self.x_positions**2 + self.y_positions**2)


class MultipleRandomWalks:
    """複数のランダムウォークをシミュレート"""

    def __init__(self, n_walkers=50, n_steps=1000, dimension=1, **kwargs):
        """
        Args:
            n_walkers: ウォーカーの数
            n_steps: 各ウォーカーのステップ数
            dimension: 次元（1または2）
            **kwargs: RandomWalk1DまたはRandomWalk2Dに渡すパラメータ
        """
        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self.dimension = dimension
        self.kwargs = kwargs
        self.walkers = []

    def simulate(self):
        """すべてのウォーカーをシミュレート"""
        self.walkers = []

        if self.dimension == 1:
            for _ in range(self.n_walkers):
                walker = RandomWalk1D(self.n_steps, **self.kwargs)
                walker.simulate()
                self.walkers.append(walker)
        else:
            for _ in range(self.n_walkers):
                walker = RandomWalk2D(self.n_steps, **self.kwargs)
                walker.simulate()
                self.walkers.append(walker)

        return self.walkers

    def plot_all(self):
        """すべてのウォーカーをプロット"""
        if not self.walkers:
            self.simulate()

        if self.dimension == 1:
            plt.figure(figsize=(12, 8))
            for walker in self.walkers:
                plt.plot(walker.positions, alpha=0.3, linewidth=0.5)
            plt.xlabel('Steps', fontsize=12)
            plt.ylabel('Position', fontsize=12)

            p_left = self.kwargs.get('p_left', 0.33)
            p_stay = self.kwargs.get('p_stay', 0.34)
            p_right = self.kwargs.get('p_right', 0.33)

            plt.title(f'{self.n_walkers} Random Walks (1D)\nLeft={p_left:.2f}, Stay={p_stay:.2f}, Right={p_right:.2f}',
                     fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)

        else:
            plt.figure(figsize=(12, 12))
            for walker in self.walkers:
                plt.plot(walker.x_positions, walker.y_positions,
                        alpha=0.4, linewidth=0.5)
                plt.scatter(walker.x_positions[-1], walker.y_positions[-1],
                           s=30, alpha=0.6)

            plt.scatter(0, 0, c='green', s=300, marker='o',
                       edgecolors='black', linewidths=2, label='Start', zorder=5)
            plt.xlabel('X Position', fontsize=12)
            plt.ylabel('Y Position', fontsize=12)

            mode = self.kwargs.get('mode', 'continuous')

            if mode == 'continuous':
                p_stay = self.kwargs.get('p_stay', 0.0)
                title_text = f'{self.n_walkers} Random Walks (2D)\nContinuous, Stay={p_stay:.2f}'
            else:
                p_right = self.kwargs.get('p_right', 0.25)
                p_up = self.kwargs.get('p_up', 0.25)
                p_left = self.kwargs.get('p_left', 0.25)
                p_down = self.kwargs.get('p_down', 0.25)
                p_stay = self.kwargs.get('p_stay', 0.0)
                title_text = f'{self.n_walkers} Random Walks (2D)\nLattice: R={p_right:.2f}, U={p_up:.2f}, L={p_left:.2f}, D={p_down:.2f}, S={p_stay:.2f}'

            plt.title(title_text, fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.legend(fontsize=10)

        plt.tight_layout()
        plt.show()

    def plot_distance_distribution(self):
        """原点からの最終距離の分布をプロット"""
        if not self.walkers:
            self.simulate()

        if self.dimension == 1:
            final_positions = [walker.positions[-1] for walker in self.walkers]
            distances = np.abs(final_positions)
        else:
            distances = [np.sqrt(walker.x_positions[-1]**2 + walker.y_positions[-1]**2)
                        for walker in self.walkers]

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(distances, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Distance from Origin', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Final Distance Distribution', fontsize=14)
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.boxplot(distances, vert=True)
        plt.ylabel('Distance from Origin', fontsize=12)
        plt.title('Distance Box Plot', fontsize=14)
        plt.grid(True, alpha=0.3)

        print(f"\n=== Statistical Summary ===")
        print(f"Mean distance: {np.mean(distances):.2f}")
        print(f"Standard deviation: {np.std(distances):.2f}")
        print(f"Median: {np.median(distances):.2f}")
        print(f"Min: {np.min(distances):.2f}, Max: {np.max(distances):.2f}")

        plt.tight_layout()
        plt.show()


def get_user_input():
    """ユーザーから設定を取得"""
    print("=" * 60)
    print("ランダムウォーク シミュレーション")
    print("Random Walk Simulation")
    print("=" * 60)

    # 次元の選択
    while True:
        dim_input = input("\n次元を選択 (1 or 2): ").strip()
        if dim_input in ['1', '2']:
            dimension = int(dim_input)
            break
        print("1 または 2 を入力してください")

    # ステップ数
    while True:
        try:
            n_steps = int(input("ステップ数 (例: 1000): ").strip())
            if n_steps > 0:
                break
            print("正の整数を入力してください")
        except ValueError:
            print("有効な整数を入力してください")

    kwargs = {}

    if dimension == 1:
        print("\n--- 1次元ランダムウォークの設定 ---")

        while True:
            try:
                p_left = float(input("左に移動する確率 (0-1, 例: 0.3): ").strip())
                p_stay = float(input("停滞する確率 (0-1, 例: 0.4): ").strip())
                p_right = float(input("右に移動する確率 (0-1, 例: 0.3): ").strip())

                total = p_left + p_stay + p_right
                if np.isclose(total, 1.0, atol=0.001):
                    kwargs['p_left'] = p_left
                    kwargs['p_stay'] = p_stay
                    kwargs['p_right'] = p_right
                    break
                else:
                    print(f"確率の合計が1になりません: {total:.3f}")
                    print("もう一度入力してください")
            except ValueError:
                print("有効な数値を入力してください")

    else:
        print("\n--- 2次元ランダムウォークの設定 ---")

        # モード選択
        while True:
            mode_input = input("移動モード (continuous/lattice, 例: continuous): ").strip().lower()
            if mode_input in ['continuous', 'lattice', 'c', 'l']:
                if mode_input == 'c':
                    mode_input = 'continuous'
                elif mode_input == 'l':
                    mode_input = 'lattice'
                kwargs['mode'] = mode_input
                break
            print("'continuous' または 'lattice' を入力してください")

        if kwargs['mode'] == 'continuous':
            # 連続角度モード：停滞確率のみ
            while True:
                try:
                    p_stay = float(input("停滞する確率 (0-1, 例: 0.2): ").strip())
                    if 0 <= p_stay <= 1:
                        kwargs['p_stay'] = p_stay
                        break
                    print("0以上1以下の値を入力してください")
                except ValueError:
                    print("有効な数値を入力してください")

        else:
            # 格子モード：各方向の確率を設定
            print("\n各方向への移動確率を設定してください")
            print("（右 + 上 + 左 + 下 + 停滞 = 1.0）")

            while True:
                try:
                    p_right = float(input("右に移動する確率 (0-1, 例: 0.25): ").strip())
                    p_up = float(input("上に移動する確率 (0-1, 例: 0.25): ").strip())
                    p_left = float(input("左に移動する確率 (0-1, 例: 0.25): ").strip())
                    p_down = float(input("下に移動する確率 (0-1, 例: 0.25): ").strip())
                    p_stay = float(input("停滞する確率 (0-1, 例: 0.0): ").strip())

                    total = p_right + p_up + p_left + p_down + p_stay
                    if np.isclose(total, 1.0, atol=0.001):
                        kwargs['p_right'] = p_right
                        kwargs['p_up'] = p_up
                        kwargs['p_left'] = p_left
                        kwargs['p_down'] = p_down
                        kwargs['p_stay'] = p_stay
                        break
                    else:
                        print(f"確率の合計が1になりません: {total:.3f}")
                        print("もう一度入力してください")
                except ValueError:
                    print("有効な数値を入力してください")

    # 複数シミュレーション
    while True:
        multi_input = input("\n複数のウォーカーをシミュレートしますか? (y/n): ").strip().lower()
        if multi_input in ['y', 'n', 'yes', 'no']:
            multiple = multi_input in ['y', 'yes']
            break
        print("y または n を入力してください")

    n_walkers = 1
    if multiple:
        while True:
            try:
                n_walkers = int(input("ウォーカーの数 (例: 50): ").strip())
                if n_walkers > 0:
                    break
                print("正の整数を入力してください")
            except ValueError:
                print("有効な整数を入力してください")

    return dimension, n_steps, kwargs, multiple, n_walkers


def run_from_config():
    """設定ファイルからパラメータを読み込んで実行"""
    print("=" * 60)
    print("ランダムウォーク シミュレーション")
    print("Random Walk Simulation")
    print("=" * 60)

    dimension = SIMULATION_CONFIG['dimension']
    n_steps = SIMULATION_CONFIG['n_steps']
    n_walkers = SIMULATION_CONFIG['n_walkers']

    # 次元に応じて設定を選択
    if dimension == 1:
        config = CONFIG_1D.copy()
        print(f"\n【設定】1次元ランダムウォーク")
        print(f"  ステップ数: {n_steps}")
        print(f"  左: {config['p_left']}, 停滞: {config['p_stay']}, 右: {config['p_right']}")
    else:
        config = CONFIG_2D.copy()
        print(f"\n【設定】2次元ランダムウォーク")
        print(f"  ステップ数: {n_steps}")
        print(f"  モード: {config['mode']}")
        if config['mode'] == 'lattice':
            print(f"  右: {config['p_right']}, 上: {config['p_up']}, 左: {config['p_left']}, 下: {config['p_down']}, 停滞: {config['p_stay']}")
        else:
            print(f"  停滞確率: {config['p_stay']}")

    if n_walkers > 1:
        print(f"  ウォーカー数: {n_walkers}")

    print("\n" + "=" * 60)
    print("シミュレーション実行中...")
    print("=" * 60)

    # シミュレーション実行
    if n_walkers > 1:
        # 複数のウォーカー
        multi = MultipleRandomWalks(n_walkers=n_walkers, n_steps=n_steps,
                                   dimension=dimension, **config)
        multi.simulate()
        multi.plot_all()
        multi.plot_distance_distribution()
    else:
        # 単一のウォーカー
        if dimension == 1:
            walker = RandomWalk1D(n_steps=n_steps, **config)
        else:
            walker = RandomWalk2D(n_steps=n_steps, **config)

        walker.simulate()
        walker.plot()

    print("\nシミュレーション完了!")


def main():
    """メイン関数"""

    # コマンドライン引数でモードを指定可能
    if len(sys.argv) > 1:
        if sys.argv[1] == 'demo':
            demo()
            return
        elif sys.argv[1] == 'interactive':
            # 対話モード
            try:
                dimension, n_steps, kwargs, multiple, n_walkers = get_user_input()

                print("\n" + "=" * 60)
                print("シミュレーション実行中...")
                print("=" * 60)

                if multiple:
                    multi = MultipleRandomWalks(n_walkers=n_walkers, n_steps=n_steps,
                                               dimension=dimension, **kwargs)
                    multi.simulate()
                    multi.plot_all()
                    multi.plot_distance_distribution()
                else:
                    if dimension == 1:
                        walker = RandomWalk1D(n_steps=n_steps, **kwargs)
                    else:
                        walker = RandomWalk2D(n_steps=n_steps, **kwargs)
                    walker.simulate()
                    walker.plot()

                print("\nシミュレーション完了!")
            except (EOFError, KeyboardInterrupt):
                print("\n\nプログラムを終了します。")
            return

    # デフォルト：設定ファイルから実行
    run_from_config()


def demo():
    """デモンストレーション"""
    print("=" * 60)
    print("Random Walk Simulation Demo")
    print("=" * 60)

    # 1. 偏りのある1次元ランダムウォーク
    print("\n1. Biased 1D Random Walk (Right-biased)")
    rw1d = RandomWalk1D(n_steps=1000, p_left=0.2, p_stay=0.3, p_right=0.5)
    rw1d.simulate()
    rw1d.plot()

    # 2. 停滞が多い2次元ランダムウォーク（連続角度）
    print("\n2. 2D Random Walk with High Stay Probability (Continuous)")
    rw2d = RandomWalk2D(n_steps=2000, p_stay=0.3, mode='continuous')
    rw2d.simulate()
    rw2d.plot()

    # 3. 格子型ランダムウォーク（等確率）
    print("\n3. Lattice Random Walk (Equal Probability)")
    rw2d_lattice = RandomWalk2D(n_steps=1000, p_stay=0.2, mode='lattice')
    rw2d_lattice.simulate()
    rw2d_lattice.plot()

    # 4. 格子型ランダムウォーク（偏りあり：右上に偏る）
    print("\n4. Lattice Random Walk (Biased to Right-Up)")
    rw2d_biased = RandomWalk2D(n_steps=1000, mode='lattice',
                               p_right=0.35, p_up=0.35, p_left=0.1, p_down=0.1, p_stay=0.1)
    rw2d_biased.simulate()
    rw2d_biased.plot()

    # 5. 複数の1次元ランダムウォーク
    print("\n5. Multiple 1D Random Walks")
    multi_1d = MultipleRandomWalks(n_walkers=50, n_steps=1000, dimension=1,
                                  p_left=0.33, p_stay=0.34, p_right=0.33)
    multi_1d.simulate()
    multi_1d.plot_all()
    multi_1d.plot_distance_distribution()

    # 6. 複数の格子型ランダムウォーク（偏りあり）
    print("\n6. Multiple Lattice Random Walks (Biased)")
    multi_2d = MultipleRandomWalks(n_walkers=30, n_steps=500, dimension=2,
                                  mode='lattice', p_right=0.3, p_up=0.3,
                                  p_left=0.15, p_down=0.15, p_stay=0.1)
    multi_2d.simulate()
    multi_2d.plot_all()
    multi_2d.plot_distance_distribution()

    print("\nDemo Complete!")


if __name__ == "__main__":
    main()
