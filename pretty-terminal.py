import time
import sys

# %% ANSI ESC
'''
\x1b[{명령어} 로 출력되는 텍스트의 스타일을 지정할 수 있다.
예를 들어, print("\x1b[1m안녕\x1b[0m")은 볼드체 "안녕"이라고 출력된다.

여러 명령어를 조합할 때는 세미콜론(;)으로 구분하여 사용할 수 있다. 
예를 들어, print("\x1b[1;31m안녕\x1b[0m")은 굵은 빨간색 "안녕"이라고 출력된다.
'''

# %% 기본 ANSI ESC 스타일
"""
0   초기화
1   굵게
2   흐리게
3   이탤릭
4   밑줄
7   반전, foreground/background 교환
9   취소선
"""
print("\x1b[1m볼드\x1b[0m", "\x1b[2m흐림\x1b[0m", "\x1b[3m이탤릭\x1b[0m", "\x1b[4m밑줄\x1b[0m", "\x1b[7m반전\x1b[0m", "\x1b[9m취소선\x1b[0m")

# %% 컬러 글자
"""
30 검정
31 빨강
32 초록
33 노랑
34 파랑
35 마젠타
36 시안
37 흰색

밝은색
90 밝은 검정, 회색
91 밝은 빨강
92 밝은 초록
93 밝은 노랑
94 밝은 파랑
95 밝은 마젠타
96 밝은 시안
97 밝은 흰색
"""

print("\x1b[32m초록\x1b[0m")
print("\x1b[31m빨\x1b[33m노\x1b[32m초\x1b[34m파\x1b[35m마\x1b[0m")

# %% 배경 색
"""
40 검정 배경
41 빨강 배경
42 초록 배경
43 노랑 배경
44 파랑 배경
45 마젠타 배경
46 시안 배경
47 흰색 배경
"""

print("\x1b[1;31m굵은 빨간 글씨\x1b[0m")
print("\x1b[1;33;44m굵은 노란 글씨 + 파란 배경\x1b[0m")

# %% RGB 컬러
'''
모던 터미널에서는 256컬러나 RGB 컬러를 지원하는 경우가 많다.
\x1b[38;2;R;G;Bm 으로 글자 색을 지정할 수 있고, \x1b[48;2;R;G;Bm 으로 배경 색을 지정할 수 있다.
'''

print("\x1b[38;2;200;100;50m화려한컬러\x1b[0m")

# %% 커서 이동
'''
\x1b[이동명령

\x1b[H       커서를 화면 맨 위 왼쪽으로 이동
\x1b[2J      화면 전체 지우기
\x1b[K       현재 줄에서 커서 오른쪽 지우기

\x1b[3A      커서를 위로 3칸 이동
\x1b[3B      커서를 아래로 3칸 이동
\x1b[3C      커서를 오른쪽으로 3칸 이동
\x1b[3D      커서를 왼쪽으로 3칸 이동
'''

print("\x1b[31m안녕\x1b[0m", end="", flush=True)
time.sleep(1)
print("\x1b[4D\x1b[3m내가 먼저지롱\x1b[0m", end="", flush=True)
time.sleep(1)
print("\x1b[8D\x1b[K", end="", flush=True)
time.sleep(1)
print("늦어", flush=True)
time.sleep(1)

# %% 커서 보이기/숨기기

'''
print("\x1b[?25l", end="")  # 커서 숨기기
print("\x1b[?25h", end="")  # 커서 보이기

다만 실행 중 프로그램이 죽으면 자동으로 커서가 복구되지 않으므로, try finally로 쓰는게 안전

try:
    print("\x1b[?25l", end="")

    적당한_함수()

finally:
    print("\x1b[?25h", end="")

'''

# %% sys.stdout

'''
print()는 평상시에는 유용함. 자동으로 줄바꿈도 해줌.
반면 sys.stdout()은 수동으로 해줘야 하는게 많지만 더 직접적인 컨트롤이 가능함.

sys.stdout.write("hello")
sys.stdout.write("world")
-> helloworld

sys.stdout.write("hello\n")
sys.stdout.write("hello\n")
-> hello
-> world

일부 환경에서는 줄바꿈이 없거나 하면 바로 출력하지 않는 경우가 있어서 수동으로 flush 해줘야 함
sys.stdout.flush()


'''

spinner = '|/-\\'

print(spinner)

for i in range(101):
    print(f"\r{spinner[i % len(spinner)]} Loading... {i} %", end="")
    time.sleep(0.03)


BAR_LEN = 20
for i in range(101):
    filled = i * BAR_LEN // 100
    bar = filled * "█" + (BAR_LEN - filled) * "░"
    print(f"\r\x1b[K{bar} {i} %", end="", flush=True)
    time.sleep(0.03)
print()