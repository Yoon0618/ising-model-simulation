from fileinput import filename

import numpy as np
import time
import matplotlib.pyplot as plt
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
            right_index[n] = (left_index[n]+1) % self.N

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
        return -self.J * 1/2 * 4 * 1 * self.N ** self.dim # 두번 샘하므로 1/2, 한 원자의 이웃은 4개이므로 4, SiSj=1일 때 최소이므로 1, 원자 개수.

    def make_state_plot_text(self):
        if self.dim == 1:
            self._state = np.reshape(self.state, (1, self.N))
        else: 
            self._state = self.state

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
        ("Status", Text("Running", style="green")),
        ("Step", f"{step}/{param['step']}"),
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
            ]
        )

    return Group(
        section_title("Status"),
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
    rng_MP = np.random.default_rng() # metropolis 방법에서 에너지가 증가했을 때 뒤집기를 유지할지 말지 결정하기 위한 랜덤 넘버 생성기.

    # 초기화
    model = State(N=param["N"], dim=param["dim"], plus_ratio=param["plus_ratio"], rng_initial_state=rng_initial_state, method=param["method"])
    # steps = np.linspace(0, param["step"])
    steps = np.arange(param["step"])
    # model.plot()

    ini_total_energy = model.eval_total_energy()
    # print(f"initial total energy = {ini_total_energy}")

    energy_history = np.empty_like(steps, dtype=np.int64)

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
                energy_history[i] = initial_energy

                index = model.random_flip()
                final_energy = model.eval_total_energy()
                energy_diff = final_energy - initial_energy

                if final_energy == model.theoretical_lowest_energy(): # 최소 에너지 도달하면 루프 종료 및 steps 슬라이싱
                    steps = steps[:i]
                    break
                
                # method에 따라 에너지가 증가했을 때 대처가 다름
                if param["method"] == "MC": # 에너지가 증가하면 변화 되돌리기
                    if energy_diff > 0:
                        model.rollback_flip(index)
                    else:
                        pass
                
                elif param["method"] == "MP": # 에너지가 증가했더라도 확률적으로 유지
                    if energy_diff > 0:
                        flip_ratio = np.exp(-param["beta"]*energy_diff)
                        coin_toss = rng_MP.choice([0, 1], size=1, p=[1-flip_ratio, flip_ratio])[0]

                        if coin_toss == 0:
                            model.rollback_flip(index)
                    else:
                        pass

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


def save_results(results):
    # shot_tracker.txt 파일이 없으면 만들고, 있으면 값 읽기
    # shot_tracker.txt에는 샷 번호를 나타내는 정수값 하나가 저장되어 있음
    try:
        with open("shot_tracker.txt", "r") as f:
            shot_number = int(f.read().strip())
    except FileNotFoundError:
        with open("shot_tracker.txt", "w") as f:
            f.write("0")
        shot_number = 0

    # npz로 결과 저장
    # 저장 성공하면 shot_tracker.txt에 저장된 샷 번호에 1 더해서 업데이트
    try:
        np.savez(f"#{shot_number}_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}.npz",
                 steps_MP = results["steps_MP"],
                 energy_history_MP = results["energy_history_MP"],
                 final_state_MP = results["final_state_MP"],
                 model_MP = results["model_MP"],
                 steps_MC = results["steps_MC"],
                 energy_history_MC = results["energy_history_MC"],
                 final_state_MC = results["final_state_MC"],
                 model_MC = results["model_MC"],
                 )
        shot_number += 1
        with open("shot_tracker.txt", "w") as f:
            f.write(str(shot_number))

    except Exception as e:
        print(f"Error occurred while saving results: {e}")

def main():
    param_MC = {
        "method": "MC",
        "dim": 2, # 차원
        "N": 20, # 그리드 개수
        "plus_ratio": 0.5, # 초기 +- 생성 비율
        "T": 1.0, # 온도, 정규화
        "beta": 1.0, # 1/kB*T, kB는 볼츠만 상수, T는 온도. 정규화 했으므로 kB=1로 생각하면 됨. 

        "step": 1000, # 총 스텝 수
        "sleep": 0, # 각 스텝마다 대기 시간 (초)
        "plot": False, # 에너지 변화 그래프 그릴지 여부

    }

    param_MP = param_MC.copy()
    param_MP["method"] = "MP"


    seed = 2026
    rng_initial_state_1 = np.random.default_rng(seed=seed)
    rng_initial_state_2 = np.random.default_rng(seed=seed)

    
    results_MC = run(param_MC, rng_initial_state=rng_initial_state_1)
    results_MP = run(param_MP, rng_initial_state=rng_initial_state_2)

    # save results
    results = {
        "steps_MC": results_MC["steps"],
        "energy_history_MC": results_MC["energy_history"],
        "final_state_MC": results_MC["final_state"],
        "N_MC": results_MC["param"]["N"],
        "dim_MC": results_MC["param"]["dim"],
        
        "steps_MP": results_MP["steps"],
        "energy_history_MP": results_MP["energy_history"],
        "final_state_MP": results_MP["final_state"],
        "N_MP": results_MP["param"]["N"],
        "dim_MP": results_MP["param"]["dim"],
    }
    save_results(results)


    # comparison of these method
    plt.plot(results_MC["steps"], results_MC["energy_history"], label="MC")
    plt.plot(results_MP["steps"], results_MP["energy_history"], label="MP")
    plt.xlabel("Steps")
    plt.ylabel("Normalized Energy")

    # theoretical lowest energy
    plt.axhline(y=results_MC["theoretical_lowest_energy"], color='r', linestyle='--', label="theoretical lowest energy")
    # 초기 에너지
    plt.axhline(y=results_MC["energy_history"][0], color='g', linestyle='--', label="initial energy")

    plt.legend()
    plt.grid()
    plt.savefig(f"comparison_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}.png", dpi=600)
    plt.show()

if __name__ == '__main__':
    main()
    
