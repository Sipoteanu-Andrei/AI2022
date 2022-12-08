import numpy as np
from utils import State
from copy import deepcopy
from dataclasses import dataclass, field

ROWS = 4
COLS = 12
START = State(3, 0)
GOAL = State(3, 11)
CLIFF = -1


@dataclass
class Cliff:
    current_pos: State = START
    board: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.board = np.zeros([4, 12])
        self.make_cliffs()

    def make_cliffs(self) -> None:
        self.board[3, 1:11] = CLIFF
        self.board[2, 2] = CLIFF
        self.board[2, 1] = CLIFF

        self.board[1, 10] = CLIFF
        self.board[2, 10] = CLIFF
        self.board[2, 7] = CLIFF

    def move(self, action: str) -> State:
        next_pos = deepcopy(self.current_pos)
        if action == "up":
            next_pos = next_pos.up()
        elif action == "down":
            next_pos = next_pos.down()
        elif action == "left":
            next_pos = next_pos.left()
        elif action == "right":
            next_pos = next_pos.right()
        else:
            raise RuntimeError("Unexpected value for action:", action)

        if 0 <= next_pos.x < ROWS and 0 <= next_pos.y < COLS:
            self.current_pos = next_pos

        return self.current_pos

    def give_reward(self) -> int:
        if self.current_pos == GOAL:
            return 0
        if self.board[self.current_pos.x, self.current_pos.y] == 0:
            return CLIFF
        return -100

    def show(self) -> None:
        show_route(self.board, pos=self.current_pos)

    @property
    def finished(self) -> bool:
        return (
            self.current_pos == GOAL
            or self.board[self.current_pos.x, self.current_pos.y] == CLIFF
        )


@dataclass
class Agent:
    exp_rate: float
    learning_rate: float
    cliff: Cliff = field(default_factory=Cliff, init=False)
    actions: list[str] = field(default_factory=list, init=False)
    pos: State = field(default=START, init=False)
    states: list = field(default_factory=list)
    state_actions: dict[State, dict[str, int]] = field(default_factory=dict)
    rewards: dict[int, float] = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.actions = ["up", "left", "right", "down"]
        if self.exp_rate != 0:
            self.cliff.show()
        for i in range(ROWS):
            for j in range(COLS):
                self.state_actions[State(i, j)] = {}
                for action in self.actions:
                    self.state_actions[State(i, j)][action] = 0

    def choose_action(self) -> str:
        if np.random.uniform(0, 1) <= self.exp_rate:
            return self.explore()
        return self.exploit()

    def explore(self) -> str:
        return np.random.choice(self.actions)

    def exploit(self) -> str:
        max_reward = -999
        action = ""

        for act in self.actions:
            current_position = self.pos
            nxt_reward = self.state_actions[current_position][act]
            if nxt_reward >= max_reward:
                action = act
                max_reward = nxt_reward

        return action

    def reset(self) -> None:
        self.states = []
        self.cliff = Cliff()
        self.pos = START

    def run(self, rounds: int) -> None:
        for round_number in range(rounds):
            while True:
                curr_state = self.pos
                cur_reward = self.cliff.give_reward()
                action = self.choose_action()

                # next position
                self.cliff.current_pos = self.cliff.move(action)
                self.pos = self.cliff.current_pos

                self.states.append([curr_state, action, cur_reward])
                if self.cliff.finished:
                    break

            reward = self.cliff.give_reward()
            for a in self.actions:
                self.state_actions[self.pos][a] = reward

            for s in reversed(self.states):
                pos, action, r = s[0], s[1], s[2]
                current_value = self.state_actions[pos][action]
                reward = current_value + self.learning_rate * (
                    r + reward - current_value
                )
                self.state_actions[pos][action] = reward
                reward = np.max(list(self.state_actions[pos].values()))
                self.rewards[round_number] = reward

            self.reset()


def show_route(
    board: np.ndarray,
    state_actions=None,
    states: list[State] = [],
    pos: State = State(-1, -1),
):
    for i in range(0, ROWS):
        print("-------------------------------------------------")
        out = "| "
        max_key = None
        max_val = None
        for j in range(0, COLS):
            if state_actions is not None:
                max_val = max([val for val in state_actions[State(i, j)].values()])
                max_key = list(state_actions[State(i, j)].keys())[
                    list(state_actions[State(i, j)].values()).index(max_val)
                ]

            token = "0"
            if board[i, j] == CLIFF:
                token = "*"
            if pos == State(-1, -1):
                if State(i, j) in states:
                    if max_key is not None and max_val is not None:
                        token = get_direction(max_key)
                    else:
                        token = "R"
            else:
                if State(i, j) == pos:
                    if max_key is not None and max_val is not None:
                        token = get_direction(max_key)
                    else:
                        token = "R"
            if State(i, j) == GOAL:
                token = "G"
            out += token + " | "
        print(out)
    print("-------------------------------------------------")


def get_direction(action: str) -> str:
    match action:
        case "up":
            return "^"
        case "down":
            return "v"
        case "left":
            return "<"
        case "right":
            return ">"
    return "_"
