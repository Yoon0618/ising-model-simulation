import rich
import time

# %% Console: Basic print
from rich.console import Console

console = Console()

console.print("[bold]Bold[/bold]")
console.print("[red]Red[/red]")
console.print("[italic blue]Italic blue[/italic blue]")

console.print("[bold green]Running[/bold green]")
console.print("[cyan]Simulation[/cyan] [yellow]started[/yellow]")

# %% Panel

from rich.panel import Panel # 패널 클래스를 콘솔.프린트 메소드의 입력으로 넣네

console.print(
    Panel("[green]Running[/green]\nProgress: 42 %",
        title="Simulation Status",
        subtitle="Ctrl+C to stop"      
          
    )
)

# %% Table

from rich.table import Table

table = Table(title="Simulation Status")

table.add_column("Name", style="cyan")
table.add_column("Value")

table.add_row("Status", "[green]Running[/green]")
table.add_row("Step", "128")
table.add_row("Progress", "33 %")
table.add_row("Spinner", "/")

console.print(table)



# %%  Progress bar: track

from rich.progress import track

for i in track(range(150), description="Simulating..."):
    time.sleep(0.01)

# %% progress bar: Progress: more detailed control

from rich.progress import Progress

import random
with Progress() as progress:
    task = progress.add_task("[cyan]Simulating...", total=100)

    for i in range(300):
        time.sleep(0.01)
        
        if random.random() > 0.5:
            progress.update(task, advance=1)

'''
정리하면 
- track()은 반복문 돌 때 자동으로 올라가기 때문에 반복문 감싸기에는 좋다.
- Progress는 진행률을 내 마음대로 설정할 수 있는게 좋다.
'''

# %% live update

# 애니메이션 예시. 쓰기만하고 지우질 않아서 실패

from rich.live import Live
from rich.console import Group

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from rich.text import Text
STEPS = 150

# [스피너] ["시뮬레이션 중..."] [진행 바] [진행률] [걸린 시간]
progress = Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"), # task_id를 지정할 때 이 태스크의 이름을 지정함. task_id = progress.add_task(description="태스트 1")
    # 이때 []은 rich 스타일도 보존하기 위함임
    BarColumn(),
    TaskProgressColumn(),
    TimeElapsedColumn(),
    console=console,
    auto_refresh=False
)

task_id = progress.add_task("[green]Simulating...[/green]", total=STEPS)


def make_dashboard(step):
    status = "Running" if step < STEPS - 1 else "Done"
    spinner = "|/-\\"[step % 4]

    # table = Table(title="Simulation Status") # 그리드 선도 있는 진짜 표
    table = Table.grid(padding=(0, 2)) # 표는 표인데, 단순히 데이터들을 정렬만 할 때. 선도 없음
    # 패딩은 두 열 간의 간격을 띄워줌

    table.add_column("Name", style="cyan")
    table.add_column("Value")

    table.add_row("Status", f"[green]{status}[/green]")
    table.add_row("Step", f"{step}")
    table.add_row("Spinner", Text(spinner, style="yellow"))

    return Panel(
        Group(
            table,
            progress,
        ),
        title="[bold cyan]Simulation progress bar[/bold cyan]",
        subtitle="[dim]Ctl+C로 중단[/dim]",
        border_style="cyan",
    )

try:
    with Live(make_dashboard(0), refresh_per_second=20, console=console) as live:
        for step in range(STEPS):
            time.sleep(0.02)

            progress.update(task_id, advance=1)
            live.update(make_dashboard(step))
            

        live.update(make_dashboard(STEPS-1))

except KeyboardInterrupt:
    console.print("[yellow]Interrupted[/yellow]")

console.print("[bold green]SIM DONE![/bold green]")

        
# %%
