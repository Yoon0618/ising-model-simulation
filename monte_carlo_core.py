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

        self.initial_energy = self.eval_total_energy()

    def counter(self):
        return {
            "plus": np.count_nonzero(self.state == 1),
            "minus": np.count_nonzero(self.state == -1),
        }

    def get_neighborhood_index(self, index):
        """
        Ising 모델의 주변부 원자들은 이웃을 2개 또는 3개만 가지게 된다.
        periodic b.c.로 모든 원자들이 4개의 이웃을 가지게 하자.
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

    def flip(self, index):
        self.state[index] *= -1
        return index

    def random_flip(self):
        random_index = tuple(self.rng.integers(0, self.N, (self.dim)))
        return self.flip(random_index)

    def rollback_flip(self, index):
        self.flip(index)
        return index

    def eval_total_energy(self):
        total_energy = 0
        for index, Si in np.ndenumerate(self.state):
            neighborhood_index = self.get_neighborhood_index(index)
            Sj = np.array([self.state[idx] for idx in neighborhood_index])
            total_energy += self.eval_energy_of_pair(Si, Sj)
        return 0.5 * total_energy

    def eval_energy_of_pair(self, Si, Sj):
        return -self.J * np.sum(Si * Sj)

    def theoretical_lowest_energy(self):
        if self.dim == 1:
            return -self.J * 1 / 2 * 2 * 1 * self.N**self.dim
        elif self.dim == 2:
            return -self.J * 1 / 2 * 4 * 1 * self.N**self.dim
        elif self.dim == 3:
            return -self.J * 1 / 2 * 6 * 1 * self.N**self.dim

        raise ValueError


def get_snapshot_steps(step_count):
    return {
        int(step_count * 0.33),
        int(step_count * 0.66),
        step_count - 2,
    }


def accept_metropolis_flip(model, beta, energy_diff, flipped_index, rng_metropolis):
    if energy_diff <= 0:
        return

    flip_ratio = np.exp(-beta * energy_diff)
    coin_toss = rng_metropolis.choice([0, 1], size=1, p=[1 - flip_ratio, flip_ratio])[0]

    if coin_toss == 0:
        model.rollback_flip(flipped_index)


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
    energy_history[0] = model.eval_total_energy()
    snapshot_steps = get_snapshot_steps(param["step"])

    last_completed_step = -1
    interrupted = False

    if on_start is not None:
        on_start(param, model)

    try:
        for i in steps:
            time.sleep(param["sleep"])

            initial_energy = model.eval_total_energy()

            flipped_index = model.random_flip()
            final_energy = model.eval_total_energy()
            energy_diff = final_energy - initial_energy

            accept_metropolis_flip(
                model=model,
                beta=param["beta"],
                energy_diff=energy_diff,
                flipped_index=flipped_index,
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
