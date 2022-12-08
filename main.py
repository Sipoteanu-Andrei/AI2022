from qlearning import *
from matplotlib import pyplot as plt


def make_graph() -> None:
    fig, ax = plt.subplots(1, 1)
    ax.bar(list(ag.rewards.keys()), [-val for val in ag.rewards.values()])
    plt.ylabel("reward")
    plt.xlabel("episode")
    plt.title("Convergence")
    plt.show()


if __name__ == "__main__":
    print("Training")
    ag = Agent(0.2, 0.1)
    ag.run(rounds=1000)

    # Q-learning
    ag_op = Agent(0, 0.1)
    ag_op.state_actions = ag.state_actions

    states: list[State] = []
    while True:
        curr_state = ag_op.pos
        action = ag_op.choose_action()
        states.append(curr_state)
        print(f"Current position {curr_state}->{action}")

        # next position
        ag_op.cliff.current_pos = ag_op.cliff.move(action)
        ag_op.pos = ag_op.cliff.current_pos

        if ag_op.cliff.finished:
            break

    board = np.zeros([ROWS, COLS])
    board[ROWS - 1, 1 : COLS - 1] = -1
    show_route(board, ag_op.state_actions,states )
    print(ag.rewards)
    make_graph()
