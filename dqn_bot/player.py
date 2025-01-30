'''
Simple example pokerbot, written in Python.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
from pokerDQN import PokerDQN, predict_action, update_model
from utils import save_model

from collections import deque
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Player(Bot):
    '''
    A pokerbot.
    '''

    def __init__(self):
        '''
        Called when a new game starts. Called exactly once.

        Arguments:
        Nothing.

        Returns:
        Nothing.
        '''
        self.model = PokerDQN(30, 8)
        self.target_model = PokerDQN(30, 8)
        self.round_counter = 0
        self.state_vectors = deque()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.1)
        self.action_id = 0
        self.round_counter = 0

        if os.path.exists("DQN_model.pth"):
            self.model.load_state_dict(torch.load("DQN_model.pth"))
            print('model loaded')

        self.target_model.load_state_dict(self.model.state_dict())
        pass

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called when a new round starts. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        self.state_vectors.clear()
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called when a round ends. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        terminal_state: the TerminalState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        my_delta = terminal_state.deltas[active]
        #print(my_delta)
        if len(self.state_vectors) > 0:
            update_model(self.model, self.target_model, self.optimizer, self.state_vectors[-1], self.action_id, my_delta, self.state_vectors[0], 0.95, True)

        self.round_counter += 1
        if self.round_counter % 50 == 0:
            self.target_model.load_state_dict(self.model.state_dict())


        if self.round_counter >= NUM_ROUNDS:
            save_model(self.model, "DQN_model.pth")


    def get_action(self, game_state, round_state, active):
        '''
        Where the magic happens - your code should implement this function.
        Called any time the engine needs an action from your bot.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Your action.
        '''
        legal_actions = round_state.legal_actions()  # the actions you are allowed to take
        street = round_state.street  # 0, 3, 4, or 5 representing pre-flop, flop, turn, or river respectively

        action, self.action_id, raise_amount = predict_action(self.model, game_state, round_state, active)
        self.state_vectors.append(self.model.preprocess_state(game_state, round_state, active))
        if street > 0:
            update_model(self.model, self.target_model, self.optimizer, self.state_vectors[-2], self.action_id, 0, self.state_vectors[-1], 0.99, False)

        #print(f'{self.state_vectors[-1].tolist()}_{self.action_id}')

        if action == "FoldAction":
            return FoldAction()
        elif action == "CheckAction" and CheckAction in legal_actions:
            return CheckAction()
        elif action == "CallAction" and CallAction in legal_actions:
            return CallAction()
        elif action == "RaiseAction" and RaiseAction in legal_actions:
            return RaiseAction(raise_amount)
        else:
            return FoldAction()


def train():
    import ast

    model = PokerDQN(30, 8)
    target_model = PokerDQN(30, 8)
    num_epoch = 6


    embeds = []
    delta = 0
    action_id = 0
    for i in range(num_epoch):
        with open('data.txt', 'r') as file:
            for line in file:
                if len(line) > 10:
                    embed, action_id = line[:-1].split('_')
                    embed = ast.literal_eval(embed)
                    action_id = int(action_id)
                    embeds.append(torch.tensor(list(embed), dtype=torch.float32))
                    if len(embeds) > 1:
                        update_model(model, target_model, optim.Adam(model.parameters(), lr=0.001), embeds[-2], action_id, embeds[-1], embeds[-1], 0.95, False)
                elif len(line) < 10:
                    delta = int(line)
                    if len(embeds) > 0:
                        update_model(model, target_model, optim.Adam(model.parameters(), lr=0.001), embeds[-1], action_id, delta, embeds[-1], 0.95, True)
        print(f'done with epoch {i+1}')

    print('saving model')
    save_model(model, "DQN_model.pth")


if __name__ == '__main__':
    #train()
    run_bot(Player(), parse_args())
