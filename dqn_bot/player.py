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
        self.model = PokerDQN(29, 8)
        self.target_model = PokerDQN(29, 8)
        self.round_counter = 0
        self.state_vectors = deque()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.action_id = 0
        self.round_counter = 0

        if os.path.exists("DQN_model.pth"):
            print("model loaded")
            self.model.load_state_dict(torch.load("DQN_model.pth"))

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
        #my_bankroll = game_state.bankroll  # the total number of chips you've gained or lost from the beginning of the game to the start of this round
        #game_clock = game_state.game_clock  # the total number of seconds your bot has left to play this game
        #round_num = game_state.round_num  # the round number from 1 to NUM_ROUNDS
        #my_cards = round_state.hands[active]  # your cards
        #big_blind = bool(active)  # True if you are the big blind
        #my_bounty = round_state.bounties[active]  # your current bounty rank

        # maybe save to have future training data

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
        #my_delta = terminal_state.deltas[active]  # your bankroll change from this round
        previous_state = terminal_state.previous_state  # RoundState before payoffs
        #street = previous_state.street  # 0, 3, 4, or 5 representing when this round ended
        #my_cards = previous_state.hands[active]  # your cards
        #opp_cards = previous_state.hands[1-active]  # opponent's cards or [] if not revealed

        my_bounty_hit = terminal_state.bounty_hits[active]  # True if you hit bounty
        opponent_bounty_hit = terminal_state.bounty_hits[1-active] # True if opponent hit bounty
        bounty_rank = previous_state.bounties[active]  # your bounty rank



        my_delta = terminal_state.deltas[active]
        if len(self.state_vectors) > 0:
            update_model(self.model, self.target_model, self.optimizer, self.state_vectors[-1], self.action_id, my_delta, self.state_vectors[0], 0.95, True)

        self.round_counter += 1
        if self.round_counter % 20 == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        if self.round_counter == NUM_ROUNDS:
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
        my_cards = round_state.hands[active]  # your cards
        board_cards = round_state.deck[:street]  # the board cards
        my_pip = round_state.pips[active]  # the number of chips you have contributed to the pot this round of betting
        opp_pip = round_state.pips[1-active]  # the number of chips your opponent has contributed to the pot this round of betting
        my_stack = round_state.stacks[active]  # the number of chips you have remaining
        opp_stack = round_state.stacks[1-active]  # the number of chips your opponent has remaining
        continue_cost = opp_pip - my_pip  # the number of chips needed to stay in the pot
        my_bounty = round_state.bounties[active]  # your current bounty rank
        my_contribution = STARTING_STACK - my_stack  # the number of chips you have contributed to the pot
        opp_contribution = STARTING_STACK - opp_stack  # the number of chips your opponent has contributed to the pot

        action, self.action_id, raise_amount = predict_action(self.model, game_state, round_state, active)
        self.state_vectors.append(self.model.preprocess_state(game_state, round_state, active))
        if street > 0:
            update_model(self.model, self.target_model, self.optimizer, self.state_vectors[-2], self.action_id, 0, self.state_vectors[-1], 0.95, False)

        if action == "FoldAction":
            return FoldAction()
        elif action == "CheckAction":
            return CheckAction()
        elif action == "CallAction":
            return CallAction()
        elif action == "RaiseAction":
            return RaiseAction(raise_amount)


if __name__ == '__main__':
    run_bot(Player(), parse_args())
