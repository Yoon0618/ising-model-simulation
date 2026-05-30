import time
import sys
from pathlib import Path

import matplotlib
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from monte_carlo_core import get_shot_number, run_simulation, save_results


console = Console()
KEY_WIDTH = 29


def make_state_plot_text(model):
    if model.dim == 1:
        state = np.reshape(model.state, (1, model.N))
    elif model.dim == 2:
        state = model.state
    elif model.dim == 3:
        raise NotImplementedError("3D state plot not implemented yet.")
    else:
        raise ValueError

    state_plot_text = ""
    for row in state:
        for cell in row:
            state_plot_text += "🟦" if cell == -1 else "🟥"
        state_plot_text += "\n"

    return state_plot_text


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
        ("Step", f"{step}/{param['step'] - 1}"),
        (
            "Ini/Current/Lowest Energy",
            f"{model.initial_energy} / {model.total_energy} / {model.theoretical_lowest_energy()}",
        ),
        (
            "plus/minus count",
            f"{model.counter()['plus']} / {model.counter()['minus']}",
        ),
    ]

    parameter_rows = [
        ("Algorithm", Text("Metropolis", style="yellow")),
        ("Dimension", str(param["dim"])),
        ("Grid size", str(param["N"])),
        ("Initial plus ratio", str(param["plus_ratio"])),
        ("Normalized Temperature (T)", str(param["T"])),
        ("Normalized Beta (1/kB*T)", str(param["beta"])),
    ]

    return Group(
        section_title(f"Shot #{param['shot_number']} Simulation Status"),
        make_kv_grid(status_rows),
        "",
        section_title("Parameter"),
        make_kv_grid(parameter_rows),
    )


def make_dashboard(step, param, model):
    left_info = make_info(step, param, model)
    right_text = Text(make_state_plot_text(model))

    body = Table.grid(expand=True, padding=(0, 2))
    body.add_column(width=60)
    body.add_column(ratio=1, overflow="fold")
    body.add_row(left_info, right_text)

    return Panel(
        body,
        title="[bold cyan]Simulation Dashboard[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    )


def make_progress(param, console):
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        auto_refresh=False,
    )
    task_id = progress.add_task(description="Simulating...", total=param["step"])

    return progress, task_id


def run(param, rng_initial_state=None, console=console):
    progress, task_id = make_progress(param, console)

    with Live(refresh_per_second=20, console=console) as live:
        def start_dashboard(_param, model):
            live.update(make_dashboard(0, _param, model))

        def update_dashboard(step, _param, model):
            progress.update(task_id, advance=1)
            live.update(make_dashboard(step, _param, model))

        results = run_simulation(
            param,
            rng_initial_state=rng_initial_state,
            on_start=start_dashboard,
            on_step=update_dashboard,
            on_snapshot=lambda model, i, _param: state_plot(model, param=_param, i=i),
        )

        live.update(make_dashboard(results["steps"][-1], param, results["model"]))

    if results["interrupted"]:
        console.print("[yellow]Interrupted[/yellow]")

    if param["plot"] is True:
        plot_energy_history(results)

    return results


def plot_energy_history(results, save_dir="./results"):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    plt.plot(results["steps"], results["energy_history"], label="Metropolis")
    plt.xlabel("Steps")
    plt.ylabel("Normalized Energy")

    plt.axhline(
        y=results["energy_history"][0],
        color="g",
        linestyle="--",
        label="initial energy",
    )
    plt.axhline(
        y=results["theoretical_lowest_energy"],
        color="r",
        linestyle="--",
        label="theoretical lowest energy",
    )

    plt.legend()
    plt.grid()

    filename = (
        save_path
        / f"#{get_shot_number()}_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}_energy_history.png"
    )
    plt.savefig(
        filename,
        dpi=600,
    )
    plt.close()


def state_plot(model, param, i=None, show=False, save_dir="./results"):
    state = model.state
    if i is None:
        filename = (
            f"{save_dir}/#{get_shot_number()}_"
            f"{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}_metropolis.png"
        )
    else:
        filename = (
            f"{save_dir}/#{get_shot_number()}_"
            f"{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}_metropolis_step-{i}.png"
        )

    plt.imshow(state, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(f"Metropolis - Step {i}" if i is not None else "Metropolis - Final State")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.savefig(f"{filename}", dpi=600)
    if show:
        plt.show()
    plt.close()


def main():
    shot_number = get_shot_number()

    param = {
        "dim": 2,
        "N": 100,
        "plus_ratio": 0.5,
        "T": 1.0,
        "beta": 1.0,
        "step": 10000,
        "sleep": 0,
        "plot": True,
        "shot_number": shot_number,
    }

    seed = 2026
    rng_initial_state = np.random.default_rng(seed=seed)

    results = run(param, rng_initial_state=rng_initial_state)

    results_to_save = {
        "steps": results["steps"],
        "energy_history": results["energy_history"],
        "final_state": results["final_state"],
        "N": results["param"]["N"],
        "dim": results["param"]["dim"],
    }

    save_results(results_to_save)


def test():
    return 0


if __name__ == "__main__":
    main()
