= 코드 흐름

초기화

루프 N 스텝
    시각화
    에너지 계산
    랜덤 자석 플립
    에너지 계산
    에너지 변화 계산
    롤백 또는 유지

최종 결과 보고

= 더 자세한 흐름

```python
import numpy as np
import time

# State class 정의
class State:
    def __init__(self, N, dim, method, plus_ratio=0.5):
        self.N = N
        self.dim = dim
        self.method = method
        self.plus_ratio = plus_ratio

        rng = np.random.default_rng(seed=2026)

        if N == 1:
            self.size = N
        elif N == 2:
            self.size = (N, N)
        elif N == 3:
            self.size = (N, N, N)
        else: 
            raise ValueError

        self.state = rng.choice([-1, +1], size=self.size, p=[1-plus_ratio, plus_ratio])

        print()

    def plot(self):
        '''
        현재 상태를 터미널 또는 matplotlib으로 시각화한다.
        '''

        # 현재 구현은 터미널 출력
        '''
        넘파이 어레이 [ -1 1 1 ] 을 string "🟦🟦🟥" 로 변환해야 한다.
        
        '''

        for row in self.state:
            print("".join("🟦" if -1 else "🟥" for row))
        


# 초기화
model = model.init(N=6, dim=1, plus_ratio=0.5, method="MC") # State 클래스?
ts = np.linspace(0, T, dt)
model.plot()



# 루프
for t in ts:
    model.plot()
    initial_energy = model.eval_energy()
    model.random_flip()
    final_energy = model.eval_energy()
    energy_difference = final_energy - initial_energy

    if energy_difference > 0:
        model.rollback_flip()
    else:
        pass

# 최종 결과
model.plot()
model.statistics()
```