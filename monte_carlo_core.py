import time

import numpy as np


class State:
    def __init__(self, N, dim, rng_initial_state=None, plus_ratio=0.5):
        self.N = N
        self.dim = dim
        self.plus_ratio = plus_ratio
        self.J = 1
        self.flip_ratio = 0.3

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
            print("INITIAL STATE: using new rng.")
            self.state = self.rng.choice(
                [-1, +1], size=self.size, p=[1 - plus_ratio, plus_ratio]
            )
        else:
            print("INITIAL STATE: using pre generated rng.")
            self.state = rng_initial_state.choice(
                [-1, +1], size=self.size, p=[1 - plus_ratio, plus_ratio]
            )

        self.neighborhood_table = self.make_neighborhood_table()
        self.total_energy = self.eval_total_energy()
        self.initial_energy = self.total_energy

    def counter(self):
        return {
            "plus": np.count_nonzero(self.state == 1),
            "minus": np.count_nonzero(self.state == -1),
        }

    def _normalize_index(self, index):
        if np.isscalar(index):
            return (int(index),)

        return tuple(index)

    def _make_neighborhood_index(self, index):
        """
        한 격자점에 대한 periodic nearest-neighbor index를 만든다.
        """
        index = np.asarray(index)
        neighborhood_index = []

        for n in range(self.dim):
            left_index = index.copy()
            right_index = index.copy()

            left_index[n] = (left_index[n] - 1) % self.N
            right_index[n] = (right_index[n] + 1) % self.N

            neighborhood_index.append(tuple(left_index))
            neighborhood_index.append(tuple(right_index))

        return neighborhood_index

    def make_neighborhood_table(self):
        """
        매 스텝 modulo 연산을 반복하지 않도록 이웃 index를 미리 저장한다.
        """
        table_shape = self.state.shape + (2 * self.dim, self.dim)
        neighborhood_table = np.empty(table_shape, dtype=np.int64)

        for index in np.ndindex(self.size):
            neighborhood_table[index] = self._make_neighborhood_index(index)

        return neighborhood_table

    def get_neighborhood_index(self, index):
        """
        Ising 모델의 주변부 원자들은 이웃을 2개 또는 3개만 가지게 된다.
        periodic b.c.로 모든 원자들이 4개의 이웃을 가지게 하자.
        """
        index = self._normalize_index(index)
        return [tuple(neighbor) for neighbor in self.neighborhood_table[index]]

    def flip(self, index, energy_diff=None):
        index = self._normalize_index(index)
        self.state[index] *= -1

        if hasattr(self, "total_energy"):
            if energy_diff is None:
                self.total_energy = self.eval_total_energy()
            else:
                self.total_energy += energy_diff

        return index

    def random_index(self):
        return tuple(self.rng.integers(0, self.N, (self.dim)))

    def random_flip(self):
        random_index = self.random_index()
        energy_diff = self.eval_energy_diff_of_flip(random_index)
        return self.flip(random_index, energy_diff=energy_diff)

    def rollback_flip(self, index, energy_diff=None):
        if energy_diff is None:
            return self.flip(index)

        return self.flip(index, energy_diff=-energy_diff)

    def eval_total_energy(self):
        """
        np.roll을 이용해 모든 nearest-neighbor pair 에너지를 한 번에 계산한다.
        """
        neighborhood_spin_sum = np.zeros_like(self.state, dtype=np.int64)

        for axis in range(self.dim):
            neighborhood_spin_sum += np.roll(self.state, 1, axis=axis)
            neighborhood_spin_sum += np.roll(self.state, -1, axis=axis)

        return -0.5 * self.J * np.sum(self.state * neighborhood_spin_sum)

    def eval_energy_diff_of_flip(self, index):
        index = self._normalize_index(index)
        neighbor_indices = self.neighborhood_table[index]
        neighbor_spins = self.state[tuple(neighbor_indices.T)]

        return 2 * self.J * self.state[index] * np.sum(neighbor_spins)

    def eval_energy_of_pair(self, Si, Sj):
        return -self.J * np.sum(Si * Sj)

    def theoretical_lowest_energy(self):
        if self.dim in {1, 2, 3}:
            return -self.J * self.dim * self.N**self.dim

        raise ValueError


def get_snapshot_steps(step_count):
    return {
        int(step_count * 0.33),
        int(step_count * 0.66),
        step_count - 2,
    }


def accept_metropolis_flip(model, beta, energy_diff, index, rng_metropolis):
    if energy_diff > 0:
        flip_ratio = np.exp(-beta * energy_diff)

        if rng_metropolis.random() >= flip_ratio:
            return False

    model.flip(index, energy_diff=energy_diff)
    return True


def run_simulation(
    param,
    rng_initial_state=None,
    on_start=None,
    on_step=None,
    on_snapshot=None,
):
    rng_metropolis = np.random.default_rng()
    model = State(
        N=param["N"],
        dim=param["dim"],
        plus_ratio=param["plus_ratio"],
        rng_initial_state=rng_initial_state,
    )

    steps = np.arange(param["step"])
    energy_history = np.empty_like(steps, dtype=np.int64)
    energy_history[0] = model.total_energy
    snapshot_steps = get_snapshot_steps(param["step"])

    last_completed_step = -1
    interrupted = False

    if on_start is not None:
        on_start(param, model)

    try:
        for i in steps:
            time.sleep(param["sleep"])

            initial_energy = model.total_energy
            flip_index = model.random_index()
            energy_diff = model.eval_energy_diff_of_flip(flip_index)

            accept_metropolis_flip(
                model=model,
                beta=param["beta"],
                energy_diff=energy_diff,
                index=flip_index,
                rng_metropolis=rng_metropolis,
            )

            energy_history[i] = initial_energy
            last_completed_step = i

            if on_snapshot is not None and i in snapshot_steps:
                on_snapshot(model, i, param)

            if on_step is not None:
                on_step(i, param, model)
    except KeyboardInterrupt:
        interrupted = True

    if interrupted:
        stop = max(last_completed_step + 1, 1)
        steps = steps[:stop]
        energy_history = energy_history[:stop]

    return {
        "model": model,
        "steps": steps,
        "energy_history": energy_history,
        "final_state": model.state,
        "theoretical_lowest_energy": model.theoretical_lowest_energy(),
        "param": param,
        "interrupted": interrupted,
    }


def save_results(results_to_save, save_dir="./results", shot_tracker_path="shot_tracker.txt"):
    shot_number = get_shot_number(shot_tracker_path)

    try:
        np.savez(
            f"{save_dir}/#{shot_number}_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}.npz",
            steps=results_to_save["steps"],
            energy_history=results_to_save["energy_history"],
            final_state=results_to_save["final_state"],
        )
        shot_number += 1
        with open(shot_tracker_path, "w") as f:
            f.write(str(shot_number))

    except Exception as e:
        print(f"Error occurred while saving results: {e}")


def get_shot_number(shot_tracker_path="shot_tracker.txt"):
    try:
        with open(shot_tracker_path, "r") as f:
            shot_number = int(f.read().strip())
    except FileNotFoundError:
        with open(shot_tracker_path, "w") as f:
            f.write("0")
        shot_number = 0

    return shot_number
