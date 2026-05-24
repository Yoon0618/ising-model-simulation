from fileinput import filename

import os
import numpy as np
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.console import Group
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.rule import Rule

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
console = Console()
SNAPSHOT_POINTS = (0, 33, 66, 99)
BETA_SWEEP_COUNT = 50
BETA_SWEEP_START = 1.0
BETA_SWEEP_STOP = 0.0001


def make_snapshot_steps(total_steps):
    return {
        0 if percent == 0 else max(1, int(np.ceil(total_steps * percent / 100))): percent
        for percent in SNAPSHOT_POINTS
    }


def make_beta_sweep_values(count=BETA_SWEEP_COUNT, start=BETA_SWEEP_START, stop=BETA_SWEEP_STOP):
    if count < 2:
        raise ValueError("Beta sweep needs at least two values.")
    if start <= 0 or stop <= 0:
        raise ValueError("Beta sweep values must be positive for geometric spacing.")
    return np.geomspace(start, stop, count)


def make_beta_color_mapper(beta_values, cmap_name="beta_temperature"):
    beta_values = np.asarray(beta_values, dtype=float)
    if np.any(beta_values <= 0):
        raise ValueError("Beta values must be positive for log-scaled colors.")

    if cmap_name == "beta_temperature":
        cmap = LinearSegmentedColormap.from_list(
            cmap_name,
            [
                "#b2182b",
                "#ef8a62",
                "#fddbc7",
                "#d1e5f0",
                "#4393c3",
                "#053061",
            ],
        )
    else:
        cmap = plt.get_cmap(cmap_name)
    norm = LogNorm(vmin=np.min(beta_values), vmax=np.max(beta_values))
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    return cmap, norm, mappable

# State class 정의
class State:
    def __init__(self, N, dim, method, rng_initial_state=None, plus_ratio=0.5):
        self.N = N
        self.dim = dim
        self.method = method
        self.plus_ratio = plus_ratio
        self.J = 1
        self.flip_ratio = 0.3

        # if method == "MC":
        #     self.flip = self.flip_MC
        # elif method == "MP":
        #     self.flip = self.flip_MP

        self.rng = np.random.default_rng()

        if self.dim == 1:
            self.size = N
        elif self.dim == 2:
            self.size = (N, N)
        elif self.dim == 3:
            self.size = (N, N, N)
        else: 
            raise ValueError

        if rng_initial_state is None:
            print(f"INITIAL STATE: using new rng.")
            self.state = self.rng.choice([-1, +1], size=self.size, p=[1-plus_ratio, plus_ratio])
        else:
            print(f"INITIAL STATE: using pre generated rng.")
            self.state = rng_initial_state.choice([-1, +1], size=self.size, p=[1-plus_ratio, plus_ratio])

        initial_plus_number = self.counter()["plus"]
        initial_minus_number = self.counter()["minus"]

        # print(f"plus count = {initial_plus_number}, minus count = {initial_minus_number}")
        # print(f"initial total energy = {self.eval_total_energy()}")

        self.initial_energy = self.eval_total_energy()

    def counter(self):
        return {
            "plus": np.count_nonzero(self.state == 1),
            "minus": np.count_nonzero(self.state == -1)
        }
    
    def get_neighborhood_index(self, index):
        '''
        Ising 모델의 주변부 원자들은 이웃을 2개 또는 3개만 가지게 된다.
        periodic b.c.로 모든 원자들이 4개의 이웃을 가지게 하자.
        '''
        index = np.asarray(index)
        neighborhood_index = []
        '''
        (0,0)
        (-1, 0) (1, 0) (0, -1) (0, 1)

        (1, 0)
        (0, 0) (2, 0) (1, -1) (1, 1)

        (N-1, N-1)
        (N-2, N-1) (N, N-1) (N-1, N-2) (N-1, N) ->  N-1) (-1, N-1) (N-1, N-2) (N-1, -1)
        즉 -1은 N-1로 바꾸고, N은 0으로 바꿔야 함
        
        '''
        for n in range(self.dim):
            left_index = index.copy()
            right_index = index.copy()

            left_index[n] = (left_index[n]-1) % self.N
            right_index[n] = (right_index[n]+1) % self.N

            neighborhood_index.append(tuple(left_index))
            neighborhood_index.append(tuple(right_index))

        return neighborhood_index

    def flip(self, index):
        self.state[index] *= -1
        return index
        
    def random_flip(self):
        random_index = tuple( self.rng.integers(0, self.N, (self.dim)) ) # 0~N-1까지의 정수 랜덤 추출
        return self.flip(random_index)

    def rollback_flip(self, index):
        self.flip(index)
        return index

    def eval_total_energy(self):
        total_energy = 0
        for index, Si in np.ndenumerate(self.state):
            neighborhood_index = self.get_neighborhood_index(index)
            # print(f"index {index}'s neighborhood: {neighborhood_index}")
            Sj = np.array([self.state[idx] for idx in neighborhood_index])
            total_energy += self.eval_energy_of_pair(Si, Sj)
        return 0.5*total_energy # 한 쌍이 두번 더해지므로 1/2 곱하기

    def eval_energy_of_pair(self, Si, Sj):
        return -self.J * np.sum(Si * Sj)

    def theoretical_lowest_energy(self):
        if self.dim == 1:
            return -self.J * 1/2 * 2 * 1 * self.N ** self.dim # 두번 샘하므로 1/2, 한 원자의 이웃은 2개이므로 2, SiSj=1일 때 최소이므로 1, 원자 개수.
        elif self.dim == 2:
            return -self.J * 1/2 * 4 * 1 * self.N ** self.dim # 두번 샘하므로 1/2, 한 원자의 이웃은 4개이므로 4, SiSj=1일 때 최소이므로 1, 원자 개수.
        elif self.dim == 3:
            return -self.J * 1/2 * 6 * 1 * self.N ** self.dim # 두번 샘하므로 1/2, 한 원자의 이웃은 6개이므로 6, SiSj=1일 때 최소이므로 1, 원자 개수.

    def make_state_plot_text(self):
        if self.dim == 1:
            self._state = np.reshape(self.state, (1, self.N))
        elif self.dim == 2:
            self._state = self.state
        elif self.dim == 3:
            raise NotImplementedError("3D state plot not implemented yet.")

        return "\n".join(
            "".join("-" if cell == -1 else "+" for cell in row)
            for row in self._state
        ) + "\n"

        state_plot_text = ""
        for row in self._state:
            for cell in row:
                state_plot_text += "🟦" if cell == -1 else "🟥"
            state_plot_text += "\n"

        return state_plot_text
    
    def plot(self):
        '''
        현재 상태를 터미널 또는 matplotlib으로 시각화한다.
        '''

        # 현재 구현은 터미널 출력
        '''
        넘파이 어레이 [ -1 1 1 ] 을 string "🟦🟦🟥" 로 변환해야 한다.
        
        '''
        if self.dim == 1:
            self._state = np.reshape(self.state, (1, self.N))
        else: 
            self._state = self.state

        print()
        for row in self._state:
            print("".join("-" if cell == -1 else "+" for cell in row))
        return

        print()
        for row in self._state:
            print("".join("🟦" if cell == -1 else "🟥" for cell in row))

KEY_WIDTH = 29

def make_kv_grid(rows):
    grid = Table.grid(expand=True, padding=(0, 1))

    grid.add_column(
        style="cyan",
        justify="right",
        width=KEY_WIDTH,
        no_wrap=True,
    )

    grid.add_column(
        ratio=1,
        overflow="fold",
    )

    for key, value in rows:
        grid.add_row(key, value)

    return grid

def section_title(title):
    return Rule(Text(title, style="bold magenta"), style="dim")

def make_info(step, param, model):
    status = "Running" if step < param["step"] - 1 else "Done"
    status_rows = [
        ("Status", Text(f"{status}", style="green")),
        ("Step", f"{step}/{param['step']-1}"),
        (
            "Ini/Current/Lowest Energy",
            f"{model.initial_energy} / {model.eval_total_energy()} / {model.theoretical_lowest_energy()}",
        ),
        (
            "plus/minus count",
            f"{model.counter()['plus']} / {model.counter()['minus']}",
        ),
    ]

    parameter_rows = [
        ("Method", Text(str(param["method"]), style="yellow")),
        ("Dimension", str(param["dim"])),
        ("Grid size", str(param["N"])),
        ("Initial plus ratio", str(param["plus_ratio"])),
    ]

    if param["method"] == "MP":
        parameter_rows.extend(
            [
                ("Normalized Temperature (T)", str(param["T"])),
                ("Normalized Beta (1/kB*T)", str(param["beta"])),
                # ("hold ratio", f"{1-np.exp(-param['beta']*():.4f}"),
            ]
        )

    return Group(
        section_title(f"Shot #{param['shot_number']} Simulation Status"),
        make_kv_grid(status_rows),
        "",
        section_title("Parameter"),
        make_kv_grid(parameter_rows),
    )

def make_dashboard(step, param, model):
    
    left_info = make_info(step, param, model)

    right_text = Text(model.make_state_plot_text())
    
    body = Table.grid(expand=True, padding=(0, 2))

    # 왼쪽은 고정 폭
    body.add_column(width=60)

    # 오른쪽은 남은 공간을 차지
    body.add_column(ratio=1, overflow="fold")

    body.add_row(left_info, right_text)

    return Panel(
        body,
        title="[bold cyan]Simulation Dashboard[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    )

def run(param, rng_initial_state=None, console=console):
    rng_MP = np.random.default_rng(param.get("mp_seed")) # metropolis 방법에서 에너지가 증가했을 때 뒤집기를 유지할지 말지 결정하기 위한 랜덤 넘버 생성기.

    # 초기화
    model = State(N=param["N"], dim=param["dim"], plus_ratio=param["plus_ratio"], rng_initial_state=rng_initial_state, method=param["method"])
    if "simulation_seed" in param:
        model.rng = np.random.default_rng(param["simulation_seed"])
    
    
    steps = np.arange(param["step"])

    # print(f"initial total energy = {ini_total_energy}")

    energy_history = np.empty_like(steps, dtype=np.int64)
    energy_history[0] = model.eval_total_energy()
    snapshot_steps = make_snapshot_steps(param["step"])
    snapshots = [
        {
            "percent": snapshot_steps[0],
            "step": 0,
            "energy": model.eval_total_energy(),
            "state": model.state.copy(),
        }
    ]
    
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        auto_refresh=False
    )

    task_id = progress.add_task(description="Simulating...", total=param["step"])

    # 루프
    try:
        with Live(make_dashboard(0, param, model), refresh_per_second=20, console=console) as live:
            for i in steps:
                time.sleep(param["sleep"])

                initial_energy = model.eval_total_energy()
                

                index = model.random_flip()
                final_energy = model.eval_total_energy()
                energy_diff = final_energy - initial_energy

                # if final_energy == model.theoretical_lowest_energy(): # 최소 에너지 도달하면 루프 종료. 이
                #     steps = steps[:i]
                #     break
                
                # method에 따라 에너지가 증가했을 때 대처가 다름
                if param["method"] == "MC": # 에너지가 증가하면 변화 되돌리기
                    if energy_diff > 0:
                        model.rollback_flip(index)
                    else:
                        pass
                
                elif param["method"] == "MP": # 에너지가 증가했더라도 확률적으로 유지
                    if energy_diff > 0:
                        hold_ratio = 1-np.exp(-param["beta"]*energy_diff)
                        coin_toss = rng_MP.choice([0, 1], size=1, p=[1-hold_ratio, hold_ratio])[0]
                        # 코인을 던져서 1이면 홀드, 0이면 롤백
                        if coin_toss == 0:
                            model.rollback_flip(index)
                    else:
                        pass

                energy_history[i] = initial_energy

                # 33%, 66%, 100% 진행될 때 상태를 이미지로 플랏해 저장
                completed_step = i + 1
                if completed_step in snapshot_steps:
                    snapshots.append(
                        {
                            "percent": snapshot_steps[completed_step],
                            "step": completed_step,
                            "energy": model.eval_total_energy(),
                            "state": model.state.copy(),
                        }
                    )

                progress.update(task_id, advance=1)
                live.update(make_dashboard(i, param, model))
            
            progress.update(task_id, advance=1)
            live.update(make_dashboard(steps[-1], param, model))
                
                
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted[/yellow]")
    

    # 최종 결과
    # model.plot()
    # print(f"final total energy = {model.eval_total_energy()}")
    # print(f"statistics plus: {model.counter()['plus']}, minus: {model.counter()['minus']}")

    # plot
    if param["plot"] == True:
        plt.plot(steps, energy_history)
        plt.xlabel("time (s)")
        plt.ylabel("Energy (J)")
        plt.savefig(f"{param['method']}_dim{param['dim']}_N{param['N']}_step{param['step']}_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}.png", dpi=600)
        plt.show()

    return {
        "model": model,
        "steps": steps,
        "energy_history": energy_history,
        "final_state": model.state,
        "snapshots": snapshots,
        "theoretical_lowest_energy": model.theoretical_lowest_energy(),
        "param": param,

    }


def test():
     # 테스트
    # model.flip((0,0))
    # model.plot()
    # print(model.eval_total_energy())

    # model.random_flip()
    # # model.flip((5, 3))
    # model.plot()
    # print(model.eval_total_energy())
    return 0


def save_results(results_to_save_MC, results_to_save_MP, save_dir="./results"):
    shot_number = get_shot_number()

    # npz로 결과 저장
    # 저장 성공하면 shot_tracker.txt에 저장된 샷 번호에 1 더해서 업데이트

    try:
        np.savez(f"{save_dir}/#{shot_number}_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}.npz",
                 steps_MC = results_to_save_MC["steps"],
                 energy_history_MC = results_to_save_MC["energy_history"],
                 final_state_MC = results_to_save_MC["final_state"],
                 
                 steps_MP = results_to_save_MP["steps"],
                 energy_history_MP = results_to_save_MP["energy_history"],
                 final_state_MP = results_to_save_MP["final_state"],
                 
                 )
        shot_number += 1
        with open("shot_tracker.txt", "w") as f:
            f.write(str(shot_number))

    except Exception as e:
        print(f"Error occurred while saving results: {e}")


def run_beta_sweep(param_template, beta_values, seed=2026, console=console):
    beta_values = np.asarray(beta_values, dtype=float)
    energy_histories = []
    steps = None
    theoretical_lowest_energy = None

    for index, beta in enumerate(beta_values, start=1):
        param = param_template.copy()
        param["method"] = "MP"
        param["beta"] = float(beta)
        param["shot_number"] = f"{param_template.get('shot_number', get_shot_number())} beta {index}/{len(beta_values)}"
        param["simulation_seed"] = seed
        param["mp_seed"] = seed + 1

        console.print(f"[cyan]Running MP beta sweep {index}/{len(beta_values)}: beta={beta:.6g}[/cyan]")
        result = run(
            param,
            rng_initial_state=np.random.default_rng(seed=seed),
            console=console,
        )

        if steps is None:
            steps = result["steps"]
            theoretical_lowest_energy = result["theoretical_lowest_energy"]
        elif not np.array_equal(steps, result["steps"]):
            raise ValueError("All beta sweep runs must use the same steps.")

        energy_histories.append(result["energy_history"])

    return {
        "steps": steps,
        "beta_values": beta_values,
        "energy_histories": np.asarray(energy_histories),
        "theoretical_lowest_energy": theoretical_lowest_energy,
        "param": param_template.copy(),
    }


def save_beta_sweep_results(
    steps,
    beta_values,
    energy_histories,
    param,
    save_dir="./results",
    shot_number=None,
):
    os.makedirs(save_dir, exist_ok=True)
    if shot_number is None:
        shot_number = get_shot_number()

    filename = (
        f"{save_dir}/#{shot_number}_"
        f"{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}_"
        "MP_beta_sweep.npz"
    )
    np.savez(
        filename,
        steps=np.asarray(steps),
        beta_values=np.asarray(beta_values),
        energy_histories_MP=np.asarray(energy_histories),
        method="MP",
        N=param["N"],
        dim=param["dim"],
        step=param["step"],
        plus_ratio=param.get("plus_ratio", np.nan),
    )
    return filename

def state_plot(model, param, i=None, show=False, save_dir="./results"):
    # state 시각화
    state = model.state
    if i is None:
        filename = f"{save_dir}/#{get_shot_number()}_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}_{param['method']}.png"
    else:
        filename = f"{save_dir}/#{get_shot_number()}_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}_{param['method']}_step-{i}.png"
    plt.imshow(state, cmap="coolwarm", vmin=-1, vmax=1)
    # plt.colorbar(label="Spin State")
    plt.title(f"{param['method']} - Step {i}" if i is not None else f"{param['method']} - Final State")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.savefig(f"{filename}", dpi=600)
    if show:
        plt.show()
    plt.close()
    
def snapshot_panel_plot(snapshots, param, show=False, save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    if len(snapshots) != len(SNAPSHOT_POINTS):
        raise ValueError(f"Expected {len(SNAPSHOT_POINTS)} snapshots, got {len(snapshots)}.")

    method_name = {
        "MC": "Monte Carlo",
        "MP": "Metropolis",
    }.get(param["method"], str(param["method"]))

    fig, axes = plt.subplots(1, len(snapshots), figsize=(14, 4.2), constrained_layout=True)
    fig.suptitle(
        f"{method_name} Spin State Snapshots (N={param['N']}, steps={param['step']})",
        fontsize=15,
        fontweight="bold",
    )

    image = None
    for ax, snapshot in zip(axes, snapshots):
        image = ax.imshow(
            snapshot["state"],
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
        )
        ax.set_title(
            f"{snapshot['percent']}% | Step {snapshot['step']}\nEnergy = {snapshot['energy']:.0f}",
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color("#444444")

    colorbar = fig.colorbar(image, ax=axes, shrink=0.72, pad=0.012)
    colorbar.set_ticks([-1, 1])
    colorbar.set_ticklabels(["-1 spin", "+1 spin"])

    filename = (
        f"{save_dir}/#{get_shot_number()}_"
        f"{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}_"
        f"{param['method']}_snapshots.png"
    )
    plt.savefig(filename, dpi=600, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return filename

def comparison_plot_results(results_MC, results_MP, show=False, save_dir="./results"):

    # comparison of these method
    plt.plot(results_MC["steps"], results_MC["energy_history"], label="Monte-Carlo")
    plt.plot(results_MP["steps"], results_MP["energy_history"], label="Metropolis")
    plt.xlabel("Steps")
    plt.ylabel("Normalized Energy")


    # 초기 에너지
    plt.axhline(y=results_MC["energy_history"][0], color='g', linestyle='--', label="initial energy")
    # theoretical lowest energy
    plt.axhline(y=results_MC["theoretical_lowest_energy"], color='r', linestyle='--', label="theoretical lowest energy")


    plt.legend()
    plt.grid()
    plt.savefig(f"{save_dir}/#{get_shot_number()}_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}.png", dpi=600)
    if show:
        plt.show()
    plt.close()


def beta_sweep_comparison_plot_results(results, show=False, save_dir="./results", shot_number=None, cmap_name="beta_temperature"):
    os.makedirs(save_dir, exist_ok=True)
    if shot_number is None:
        shot_number = get_shot_number()

    steps = results["steps"]
    beta_values = results["beta_values"]
    energy_histories = results["energy_histories"]

    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    cmap, norm, mappable = make_beta_color_mapper(beta_values, cmap_name=cmap_name)

    for beta, energy_history in zip(beta_values, energy_histories):
        ax.plot(
            steps,
            energy_history,
            color=cmap(norm(beta)),
            linewidth=1.0,
            alpha=0.85,
        )

    ax.set_title("Metropolis Energy History by Beta")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Energy")
    ax.axhline(
        y=energy_histories[0][0],
        color="g",
        linestyle="--",
        linewidth=1,
        label="initial energy",
    )
    ax.axhline(
        y=results["theoretical_lowest_energy"],
        color="r",
        linestyle="--",
        linewidth=1,
        label="theoretical lowest energy",
    )
    ax.grid(True, alpha=0.3)

    colorbar = fig.colorbar(mappable, ax=ax, pad=0.02)
    colorbar.set_label("beta")

    color_tag = "" if cmap_name == "beta_temperature" else f"_{cmap_name}"
    filename = (
        f"{save_dir}/#{shot_number}_"
        f"{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}_"
        f"MP_beta_sweep{color_tag}.png"
    )
    plt.savefig(filename, dpi=600, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return filename

def get_shot_number():
    # shot_tracker.txt 파일이 없으면 만들고, 있으면 값 읽기
    # shot_tracker.txt에는 샷 번호를 나타내는 정수값 하나가 저장되어 있음
    try:
        with open("shot_tracker.txt", "r") as f:
            shot_number = int(f.read().strip())
    except FileNotFoundError:
        with open("shot_tracker.txt", "w") as f:
            f.write("0")
        shot_number = 0

    return shot_number


def main():
    # read shot number from shot_tracker.txt, if not exist, create one and initialize with 0
    shot_number = get_shot_number()
    save_dir = "./results"

    param_MC = {
        "method": "MC",
        "dim": 2, # 차원
        "N": 20, # 그리드 개수
        "plus_ratio": 0.5, # 초기 +- 생성 비율
        "T": 1.0, # 온도, 정규화
        "beta": 0.05, # 1/kB*T, kB는 볼츠만 상수, T는 온도. 정규화 했으므로 kB=1로 생각하면 됨. 

        "step": 10000, # 총 스텝 수
        "sleep": 0, # 각 스텝마다 대기 시간 (초)
        "plot": False, # 에너지 변화 그래프 그릴지 여부
        "shot_number": shot_number, # 샷 번호, 결과 저장할 때 파일 이름에 포함됨

    }

    param_MP = param_MC.copy()
    param_MP["method"] = "MP"


    seed = 2026
    beta_values = make_beta_sweep_values()
    beta_sweep_results = run_beta_sweep(
        param_MP,
        beta_values,
        seed=seed,
        console=console,
    )
    save_beta_sweep_results(
        steps=beta_sweep_results["steps"],
        beta_values=beta_sweep_results["beta_values"],
        energy_histories=beta_sweep_results["energy_histories"],
        param=param_MP,
        save_dir=save_dir,
        shot_number=shot_number,
    )
    beta_sweep_comparison_plot_results(
        beta_sweep_results,
        show=False,
        save_dir=save_dir,
        shot_number=shot_number,
    )
    with open("shot_tracker.txt", "w") as f:
        f.write(str(shot_number + 1))
    return

    rng_initial_state_1 = np.random.default_rng(seed=seed)
    rng_initial_state_2 = np.random.default_rng(seed=seed)

    
    results_MC = run(param_MC, rng_initial_state=rng_initial_state_1)
    results_MP = run(param_MP, rng_initial_state=rng_initial_state_2)

    snapshot_panel_plot(results_MC["snapshots"], param_MC, show=False, save_dir=save_dir)
    snapshot_panel_plot(results_MP["snapshots"], param_MP, show=False, save_dir=save_dir)

    # plot results
    comparison_plot_results(results_MC, results_MP, show=False, save_dir=save_dir)

    # save results
    results_to_save_MC = {
        "steps": results_MC["steps"],
        "energy_history": results_MC["energy_history"],
        "final_state": results_MC["final_state"],
        "N": results_MC["param"]["N"],
        "dim": results_MC["param"]["dim"],
    }

    results_to_save_MP = {
        "steps": results_MP["steps"],
        "energy_history": results_MP["energy_history"],
        "final_state": results_MP["final_state"],
        "N": results_MP["param"]["N"],
        "dim": results_MP["param"]["dim"],
    }

    save_results(results_to_save_MC, results_to_save_MP)


    

if __name__ == '__main__':
    main()
    
