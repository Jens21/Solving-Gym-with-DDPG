from pyvirtualdisplay import Display
import argparse

from network import Network
from environment import Environment

N_TOTAL = 1_000

def main():
    env = Environment(n_envs=1, render_mode='human')
    network = Network(load=True)

    obs = env.reset()
    for itt in range(N_TOTAL):
        action = network.get_network_action(obs)
        obs, _, _ = env.step(action)

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--display', action="store_true", help='a flag indicating whether training runs in a virtual environment')

    args = parser.parse_args()

    if args.display:
        display = Display(visible=0, size=(800, 600))
        display.start()
        print('Display started')

    main()

    if args.display:
        display.stop()
