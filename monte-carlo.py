import numpy as np
import time
import matplotlib.pyplot as plt

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

        print(f"plus count = {initial_plus_number}, minus count = {initial_minus_number}")
        print(f"initial total energy = {self.eval_total_energy()}")

        
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

def run(method="MP", rng_initial_state=None):
    # 파라미터
    # method = "MP" # 수치 방법 설정. "MC" or "MP"
    dim = 2 # 차원
    N = 20 # 그리드 개수
    plus_ratio = 0.5 # 초기 +- 생성 비율
    T = 1 # 온도, 정규화
    beta = 1.0 # 1/kB*T, kB는 볼츠만 상수, T는 온도. 정규화 했으므로 kB=1로 생각하면 됨. 
    
    step = 50000 # 총 스텝 수
    ts = np.arange(0, step)
    plot = False
    rng_MP = np.random.default_rng()

    # 초기화
    model = State(N=N, dim=dim, plus_ratio=plus_ratio, rng_initial_state=rng_initial_state, method=method)
    model.plot()

    ini_total_energy = model.eval_total_energy()
    print(f"initial total energy = {ini_total_energy}")

    energy_history = np.empty_like(ts, dtype=np.int64)

    # 루프
    for i, t in enumerate(ts):
        model.plot()
        initial_energy = model.eval_total_energy()
        energy_history[i] = initial_energy

        index = model.random_flip()
        final_energy = model.eval_total_energy()
        energy_diff = final_energy - initial_energy

        if final_energy == model.theoretical_lowest_energy(): # 최소 에너지 도달하면 루프 종료 및 ts 슬라이싱
            ts = ts[:i]
            break
        
        # method에 따라 에너지가 증가했을 때 대처가 다름
        if method == "MC": # 에너지가 증가하면 변화 되돌리기
            if energy_diff > 0:
                model.rollback_flip(index)
            else:
                pass
        
        elif method == "MP": # 에너지가 증가했더라도 확률적으로 유지
            if energy_diff > 0:
                flip_ratio = np.exp(-beta*energy_diff)
                coin_toss = rng_MP.choice([0, 1], size=1, p=[1-flip_ratio, flip_ratio])[0]

                if coin_toss == 0:
                    model.rollback_flip(index)
            else:
                pass
        

        # time.sleep(0.1)

    # 최종 결과
    model.plot()
    print(f"final total energy = {model.eval_total_energy()}")
    print(f"statistics plus: {model.counter()['plus']}, minus: {model.counter()['minus']}")

    # plot
    if plot == True:
        plt.plot(ts, energy_history)
        plt.xlabel("time (s)")
        plt.ylabel("Energy (J)")
        plt.savefig(f"{method}_dim{dim}_N{N}_step{T}_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}.png", dpi=600)
        plt.show()

    return {
        "model": model,
        "ts": ts,
        "energy_history": energy_history,
        "final_state": model.state,
        "theoretical_lowest_energy": model.theoretical_lowest_energy(),

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

def main():
    seed = 2026
    rng_initial_state_1 = np.random.default_rng(seed=seed)
    rng_initial_state_2 = np.random.default_rng(seed=seed)

    results_MC = run(method="MC", rng_initial_state=rng_initial_state_1)
    results_MP = run(method="MP", rng_initial_state=rng_initial_state_2)

    # comparison of these method
    plt.plot(results_MC["ts"], results_MC["energy_history"], label="MC")
    plt.plot(results_MP["ts"], results_MP["energy_history"], label="MP")
    plt.xlabel("time (s)")
    plt.ylabel("Energy (J)")

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
    
