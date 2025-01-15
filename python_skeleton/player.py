'''
Simple example pokerbot, written in Python.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random


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

        # The following is a demonstration of accessing illegal information (will not work)
        opponent_bounty_rank = previous_state.bounties[1-active]  # attempting to grab opponent's bounty rank

        if my_bounty_hit:
            print("I hit my bounty of " + bounty_rank + "!")
        if opponent_bounty_hit:
            print("Opponent hit their bounty of " + opponent_bounty_rank + "!")

    def get_hand_strenth(self, my_cards, board_cards):
        '''
        Called when deciding action, gauges strength of current hand.

        Arguments: my_cards -> player-specific cards board_cards -> community cards

        Returns:
        strong: current hand is good, medium: current hand is OK, weak: current hand is bad
        '''

        # rank and suit variables
        board_ranks = [card[0] for card in board_cards]
        board_suits = [card[1] for card in board_cards]
        my_ranks = [card[0] for card in my_cards]
        my_suits = [card[1] for card in my_cards]
        total_ranks = board_ranks + my_ranks
        total_suits = board_suits + my_suits

        # define functions for checking the type of hand
        # royal flush
        def check_royal_flush(total_ranks, total_suits):
            suit = total_suits[0]
            for other_suit in total_suits[1:]:
                if other_suit != suit:
                    return False
            for letter in 'AKQJT':
                if letter not in total_ranks:
                    return False
            return True
        # straight flush
        def check_straight_flush(total_ranks, total_suits):
            if check_flush(total_ranks, total_suits) and check_straight(total_ranks, total_suits):
                return True
            return False
        # four of a kind
        def check_quads(total_ranks, total_suits):
            equal_rank = max(total_ranks.count(total_ranks[0]), total_ranks.count(total_ranks[1]))
            if equal_rank == 4:
                return True
            return False
        # full house
        def check_full_house(total_ranks, total_suits):
            total_ranks.sort()
            if total_ranks[0] == total_ranks[1]:
                if total_ranks[3] == total_ranks[4]:
                    if total_ranks[2] == total_ranks[0] or total_ranks[2] == total_ranks[4]:
                        return True
            return False
        # flush
        def check_flush(total_ranks, total_suits):
            for suit in total_suits:
                if suit != total_suits[0]:
                    return False
            return True
        # straight
        def check_straight(total_ranks, total_suits):
            ranks = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12,
                        'K': 13, 'A': 14}

            if '2' in total_ranks:
                ranks['A'] = 1

            total_ranks = sorted(total_ranks, key=lambda item: ranks[item])
            for i in range(len(total_ranks[1:])):
                if ranks[total_ranks[i]] != ranks[total_ranks[i-1]] + 1:
                    return False
            return True
        # three of a kind
        def check_trips(total_ranks, total_suits):
            total_ranks.sort()
            # after sorting, trips can either be first 3 or last 3 elements
            if total_ranks[0] == total_ranks[1]:
                if total_ranks[0] == total_ranks[2]:
                    return True
            if total_ranks[2] == total_ranks[3]:
                if total_ranks[3] == total_ranks[4]:
                    return True
            return False
        # two pair
        def check_two_pair(total_ranks, total_suits):
            unique = set()
            for rank in total_ranks:
                unique.add(rank)
            if len(unique) == 3:
                return True
            return False
        # pair
        def check_pair(total_ranks, total_suits):
            unique = set()
            for rank in total_ranks:
                unique.add(rank)
            if len(unique) == 4:
                return True
            return False

        # pre-flop logic
        if len(board_cards) == 0:
            ranks = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12,
                        'K': 13, 'A': 14}

            rank_sum = ranks[my_cards[0][0]] + ranks[my_cards[1][0]]
            if my_cards[0][0] in 'AKQJT' and my_cards[1][0] in 'AKQJT':
                return "strong"
            elif rank_sum >= 19:
                return "medium"
            else:
                return "weak"

        # flop logic
        elif len(board_cards) == 3:

            if check_royal_flush(total_ranks, total_suits):
                return "strong"

            elif check_straight_flush(total_ranks, total_suits):
                return "strong"

            elif check_quads(total_ranks, total_suits):
                return "strong"

            elif check_full_house(total_ranks, total_suits):
                return "strong"

            elif check_flush(total_ranks, total_suits):
                return "strong"

            elif check_straight(total_ranks, total_suits):
                return "strong"

            elif check_trips(total_ranks, total_suits):
                return "strong"

            elif check_two_pair(total_ranks, total_suits):
                return "strong"

            elif check_pair(total_ranks, total_suits):
                pair_rank = ""
                unique = set()
                for rank in total_ranks:
                    if rank in unique:
                        pair_rank = rank
                        break
                    unique.add(rank)

                if ord(pair_rank) - 48 >= 9:
                    return "strong"
                return "medium"

            else:
                return "weak"

        # turn logic
        elif len(board_cards) == 4:
            if check_royal_flush(total_ranks, total_suits):
                return "strong"

            elif check_straight_flush(total_ranks, total_suits):
                return "strong"

            elif check_quads(total_ranks, total_suits):
                return "strong"

            elif check_full_house(total_ranks, total_suits):
                return "strong"

            elif check_flush(total_ranks, total_suits):
                return "strong"

            elif check_straight(total_ranks, total_suits):
                return "strong"

            elif check_trips(total_ranks, total_suits):
                return "strong"

            elif check_two_pair(total_ranks, total_suits):
                return "strong"

            elif check_pair(total_ranks, total_suits):
                pair_rank = ""
                unique = set()
                for rank in total_ranks:
                    if rank in unique:
                        pair_rank = rank
                        break
                    unique.add(rank)

                if ord(pair_rank) - 48 >= 9:
                    return "medium"
                return "weak"

            else:
                return "weak"

        # river logic
        else:
            if check_royal_flush(total_ranks, total_suits):
                return "strong"

            elif check_straight_flush(total_ranks, total_suits):
                return "strong"

            elif check_quads(total_ranks, total_suits):
                return "strong"

            elif check_full_house(total_ranks, total_suits):
                return "strong"

            elif check_flush(total_ranks, total_suits):
                return "strong"

            elif check_straight(total_ranks, total_suits):
                return "strong"

            elif check_trips(total_ranks, total_suits):
                return "strong"

            elif check_two_pair(total_ranks, total_suits):
                return "medium"

            elif check_pair(total_ranks, total_suits):
                pair_rank = ""
                unique = set()
                for rank in total_ranks:
                    if rank in unique:
                        pair_rank = rank
                        break
                    unique.add(rank)

                if pair_rank in "AKQJ":
                    return "medium"
                return "weak"

            else:
                return "weak"

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




        strength = self.get_hand_strenth(my_cards, board_cards)

        # decide weather to RaiseAction, CallAction, CheckAction, or FoldAuction
        if strength == 'strong':
            if RaiseAction in legal_actions:
                min_raise, max_raise = round_state.raise_bounds()
                raise_amount = int(min_raise + (max_raise - min_raise)*random.uniform(0, 1))

                if my_cards[0][0] == my_bounty or my_cards[1][0] == my_bounty:
                    return RaiseAction(raise_amount)
                if random.random() < .8:
                    multiplier = random.random()
                    return RaiseAction(raise_amount * multiplier)

            if CallAction in legal_actions:
                if random.random() < .95:
                    return CallAction()

            if CheckAction in legal_actions:
                return CheckAction()
            else:
                return FoldAction()

        elif strength == 'medium':
            if RaiseAction in legal_actions:
                min_raise, max_raise = round_state.raise_bounds()
                raise_amount = int(min_raise + (max_raise - min_raise) * random.uniform(0, 1) * .5)

                if (my_cards[0][0] == my_bounty or my_cards[1][0] == my_bounty) and random.random() < 0.4:
                    return RaiseAction(raise_amount)
                elif random.random() < 0.2:
                    return RaiseAction(raise_amount)

            if CallAction in legal_actions:
                if random.random() < 0.8:
                    return CallAction()

            if CheckAction in legal_actions:
                return CheckAction()
            else:
                return FoldAction()

        elif strength == 'weak':
            if RaiseAction in legal_actions:
                min_raise, max_raise = round_state.raise_bounds()
                raise_amount = int(min_raise + (max_raise - min_raise) * random.uniform(0, 1))

                if random.random() < 0.1:
                    return RaiseAction(raise_amount)

            if CallAction in legal_actions:
                if random.random() < 0.05:
                    return CallAction()
            if CheckAction in legal_actions:
                return CheckAction()
            else:
                return FoldAction()
        else:
            return FoldAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
