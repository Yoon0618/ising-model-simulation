import sys
import time
import shutil

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
RED = "\x1b[31m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
CYAN = "\x1b[36m"
DIM = "\x1b[2m"

HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"
CLEAR_SCREEN = "\x1b[2J"
HOME = "\x1b[H"

STEPS = 500
PROGRESS_BAR_WIDTH = 20
BOX_WIDTH = 50

def make_row(content):
    spacing = BOX_WIDTH - len(content)
    return "│ " + content + " " * spacing + " │"

def make_top():
    spacing = BOX_WIDTH
    return "┌─" + "─" * spacing + "─┐"

def make_bottom():
    spacing = BOX_WIDTH
    return "└─" + "─" * spacing + "─┘"

def make_progress_bar(percent):
    '''
    0 % -> 0
    100 % -> 20
    1 % -> 0
    5 % -> 1
    '''

    filled = round(percent / 100 * PROGRESS_BAR_WIDTH) # round(percent/100) * 너비

    progress_bar = "█" * filled + "░" * (PROGRESS_BAR_WIDTH - filled) 
    
    return progress_bar + f" {percent:>6.2f} %"

def make_frame(i, status):
    percent = i / (STEPS-1) * 100 # i=0 -> 0%, i=999 -> 100%
    columns, rows = shutil.get_terminal_size()
    spinner = "|/-\\"[i % 4]

    content_row1 = f"{"Status":<8} : {GREEN}{status}{RESET}"
    content_row2 = f"{"Progress":<8} : {GREEN}{make_progress_bar(percent)}{RESET}"
    content_row3 = f"{"Spinner":<8} : {YELLOW}{spinner}{RESET}"

    # \을 붙이면 엔터 쳐도 줄바꿈 X
    frame = f"""\
    {BOLD}{CYAN}{"Simulation Progess Bar":^}{RESET}
    {DIM}terminal size: {columns}x{rows}{RESET}
    0         1         2         3         4         5      
    012345678901234567890123456789012345678901234567890123456789
    {make_top()}
    {make_row(content_row1)}
    {make_row(content_row2)}
    {make_row(content_row3)}
    {make_bottom()}

    {DIM}Ctl + C 로 중단{RESET}

    """

    return frame

try:
    # 화면 지우고 커서 숨기기
    sys.stdout.write(CLEAR_SCREEN)
    sys.stdout.write(HIDE_CURSOR)

    # 반복문으로 화면 계속 업데이트
    for i in range(STEPS):
        sys.stdout.write(HOME)
        sys.stdout.write(make_frame(i, status="Running"))
        sys.stdout.flush()
        time.sleep(0.05)
    sys.stdout.write(HOME)
    sys.stdout.write(make_frame(STEPS-1, status="Done"))
    sys.stdout.flush()

except KeyboardInterrupt:
    pass


finally:
    # 커서 복원
    sys.stdout.write(SHOW_CURSOR)
    sys.stdout.write(RESET)
    sys.stdout.write("\n")
    sys.stdout.flush()

print(f"SIM DONE!")