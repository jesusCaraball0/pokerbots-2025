from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random
import eval7
import math
import time
import numpy as np


RATINGS = {17563648: (100.5499, 1), 17498112: (92.6761, 2), 17432576: (85.5096, 3), 17367040: (78.41845, 4), 17301504: (71.4725, 5), 17235968: (63.1444, 6), 17170432: (54.79815, 7), -831488: (48.45775, 8), 17104896: (46.35785, 9), -827392: (46.19175, 10), -823296: (43.84875, 11), 831488: (43.73325, 12), -819200: (41.4161, 13), 827392: (41.1587, 14), 823296: (39.0159, 15), 17039360: (38.54785, 16), -761856: (38.28495, 17), -815104: (36.5618, 18), -757760: (36.1625, 19), 819200: (35.9326, 20), -811008: (34.0538, 21), -753664: (33.74195, 22), 761856: (32.86195, 23), -806912: (31.6492, 24), 815104: (30.5085, 25), 757760: (29.90955, 26), 16973824: (29.64005, 27), -692224: (29.3029, 28), -749568: (28.59835, 29), -798720: (28.44875, 30), -802816: (28.4405, 31), 753664: (27.863, 32), 811008: (27.76565, 33), -688128: (27.0743, 34), -794624: (25.93965, 35), 806912: (25.4826, 36), -790528: (23.78695, 37), -745472: (23.65495, 38), 692224: (22.9779, 39), 749568: (22.4367, 40), -684032: (22.3795, 41), -741376: (22.05445, 42), -622592: (22.0165, 43), 802816: (21.582, 44), 798720: (21.33945, 45), 688128: (21.1805, 46), -786432: (20.78945, 47), 16908288: (19.8143, 48), -737280: (19.0773, 49), 794624: (19.04265, 50), -679936: (17.26615, 51), 745472: (16.77115, 52), -618496: (16.5341, 53), 790528: (16.42795, 54), -733184: (16.3108, 55), 684032: (15.2581, 56), 622592: (14.63495, 57), 741376: (14.60415, 58), 786432: (14.014, 59), -729088: (13.8193, 60), -675840: (12.27215, 61), 737280: (12.1693, 62), -614400: (11.6996, 63), -724992: (11.638, 64), -552960: (11.40535, 65), 16842752: (10.7767, 66), 679936: (10.19205, 67), -671744: (10.16125, 68), 618496: (9.62335, 69), 733184: (9.34945, 70), -720896: (9.2279, 71), -667648: (7.99205, 72), -610304: (6.9531, 73), 729088: (6.49935, 74), -548864: (6.22765, 75), -663552: (6.1325, 76), 675840: (5.2547, 77), 614400: (4.51715, 78), 552960: (4.3659, 79), 724992: (3.9622, 80), -659456: (3.16415, 81), 671744: (3.12565, 82), -483328: (2.43485, 83), -544768: (1.90685, 84), -606208: (1.3464, 85), 720896: (1.19955, 86), 16777216: (1.12035, 87), -655360: (0.79915, 88), -602112: (0.22055, 89), 667648: (-0.11385, 90), 548864: (-0.9119, 91), 610304: (-1.1374, 92), -479232: (-2.32485, 93), -598016: (-2.6213, 94), 663552: (-2.6763, 95), -540672: (-3.03655, 96), -593920: (-5.0446, 97), 659456: (-5.22555, 98), 483328: (-5.69195, 99), 544768: (-5.9928, 100), 606208: (-6.10115, 101), -413696: (-6.10775, 102), -475136: (-7.5504, 103), -536576: (-7.89525, 104), -589824: (-7.91505, 105), 655360: (-8.063, 106), 602112: (-8.1037, 107), -532480: (-10.0606, 108), 598016: (-10.3642, 109), 479232: (-10.50115, 110), -409600: (-10.85095, 111), 540672: (-11.52745, 112), -471040: (-12.36455, 113), -344064: (-12.99815, 114), -528384: (-13.06305, 115), 593920: (-13.75715, 116), 413696: (-14.3297, 117), -524288: (-14.44905, 118), 475136: (-16.01215, 119), -405504: (-16.1161, 120), 536576: (-16.16065, 121), 589824: (-16.29925, 122), -466944: (-18.00315, 123), -339968: (-18.1929, 124), 532480: (-18.6626, 125), -274432: (-19.2654, 126), 409600: (-19.3281, 127), -462848: (-19.61355, 128), -401408: (-20.8428, 129), 471040: (-20.9462, 130), 528384: (-21.20745, 131), 344064: (-21.52645, 132), -458752: (-21.5325, 133), 524288: (-23.82765, 134), -204800: (-23.87275, 135), -335872: (-23.92115, 136), 405504: (-23.97395, 137), -270336: (-24.76485, 138), -397312: (-26.2592, 139), 466944: (-26.68985, 140), 339968: (-27.3174, 141), -393216: (-27.88995, 142), -331776: (-28.19135, 143), 462848: (-28.40365, 144), 274432: (-28.44325, 145), -200704: (-29.403, 146), 401408: (-30.0234, 147), -266240: (-30.4744, 148), 458752: (-31.3863, 149), -135168: (-32.67935, 150), 335872: (-33.3971, 151), -327680: (-33.58245, 152), 270336: (-34.01255, 153), 204800: (-34.2243, 154), -196608: (-34.50315, 155), -262144: (-34.804, 156), 397312: (-35.5102, 157), -131072: (-37.6244, 158), 393216: (-37.62605, 159), 331776: (-38.104, 160), 266240: (-39.3327, 161), 200704: (-39.65555, 162), -65536: (-40.31225, 163), 135168: (-42.2411, 164), 327680: (-43.65515, 165), 196608: (-44.6336, 166), 262144: (-45.42615, 167), 131072: (-48.20695, 168), 65536: (-50.48615, 169)}
def rate_hand(hand):
    return RATINGS[eval7.evaluate(hand)]


class Player(Bot):
    '''
    woof
    '''


    def __init__(self):
        # stats
        self.fold_counter = 0
        self.chop_counter = 0
        self.call_win_counter = {}
        self.time_thinking = 0

        # game states
        self.auto_fold = False
        self.opp_projected_win = False

        # round states
        self.opp_bounty = None
        self.opp_bounty_pos = set(range(0, 13))
        self.preflop_strength = None


    def handle_new_round(self, game_state, round_state, active):
        round_num = game_state.round_num
        my_bankroll = game_state.bankroll
        opp_bankroll = -game_state.bankroll

        my_cards = round_state.hands[active]
        my_cards = [eval7.Card(card) for card in my_cards]


        # strategic fold and tank blinds
        rounds_left = NUM_ROUNDS - round_num + 1
        rotations_left = math.floor(rounds_left / 2)
        current_blind = rounds_left % 2 == 1 and (bool(active) and BIG_BLIND or SMALL_BLIND) or 0
        blind_cost = current_blind + (SMALL_BLIND + BIG_BLIND) * rotations_left
        bounty_cost = blind_cost * .5 + 10 * rounds_left # worst case scenario where opp always hits bounty

        # worst case scenario chance that opp hits bounty
        total_prob = 0
        for pos in self.opp_bounty_pos:
            i=0#amount of opp bounty cards in my hand
            for card in my_cards:
                if card.rank==pos:
                    i+=1
            prob=(4-i)/50+(50-4+i)*(4-i)/(50*49)
            total_prob+=prob/len(self.opp_bounty_pos)
        bounty_rate=total_prob * 1.1
        max_payment = blind_cost + math.ceil(bounty_cost)
        remaining_payment = blind_cost + math.ceil(bounty_cost * bounty_rate) # fold if above this threshold

        if my_bankroll - remaining_payment > 20:
            if not self.auto_fold:
                self.auto_fold = True
                print(f"STRATEGIC FOLD @ {round_num}, ${my_bankroll}\t\t(${remaining_payment}, MAX ${max_payment})")

        # opp will probably win, play more risky before they start tanking blinds
        if opp_bankroll > blind_cost * .8:
            self.opp_projected_win = True


    def handle_round_over(self, game_state, terminal_state, active):
        round_num = game_state.round_num
        my_delta = terminal_state.deltas[active]  # your bankroll change from this round
        previous_state = terminal_state.previous_state
        opponent_bounty_hit = terminal_state.bounty_hits[1 - active]
        street = previous_state.street
        opp_cards = previous_state.hands[1 - active]
        board_cards = previous_state.deck[:street]
        my_contrib = STARTING_STACK - previous_state.stacks[active]
        opp_contrib = STARTING_STACK - previous_state.stacks[1 - active]


        # discern opponent bounty
        if round_num % 25 == 0: # bounties reset every 25 rounds
            self.opp_bounty = None
            self.opp_bounty_pos = set(range(0, 13))
        elif len(self.opp_bounty_pos) > 0: # bounty has yet to be determined
            pos = set([eval7.Card(card).rank for card in opp_cards + board_cards])
            if opponent_bounty_hit:
                if opp_cards:
                    self.opp_bounty_pos.intersection_update(pos)
            elif my_delta <= 0:
                self.opp_bounty_pos.difference_update(pos)
            if len(self.opp_bounty_pos) == 1: # bounty determined
                self.opp_bounty = self.opp_bounty_pos.pop()


        # record stats
        if my_delta == 0:
            self.chop_counter += 1

        if my_contrib > (bool(active) and BIG_BLIND or SMALL_BLIND):
            if my_delta > 0:
                self.call_win_counter[round_num] = 1
            else:
                self.call_win_counter[round_num] = 0

        if my_contrib < opp_contrib and not self.auto_fold:
            self.fold_counter += 1


        # match over
        if round_num == NUM_ROUNDS:
            call_win_counter = sum(self.call_win_counter.values())
            num_calls = len(self.call_win_counter)

            print("\nStats:")
            print("Fold %:", self.fold_counter / NUM_ROUNDS * 100, self.fold_counter)
            print("Chop %:", self.chop_counter / NUM_ROUNDS * 100, self.chop_counter)
            print("Call Win %:", call_win_counter / num_calls * 100, call_win_counter, num_calls)
            print("Time Spent Thinking (s):", round(self.time_thinking, 2))
            if self.auto_fold:
                print("\nAUTO FOLDED" * 10)


    def get_action(self, game_state, round_state, active):
        legal_actions = round_state.legal_actions()
        round_num = game_state.round_num
        street = round_state.street
        my_cards = round_state.hands[active]
        board_cards = round_state.deck[:street]
        my_bounty = eval7.ranks.index(round_state.bounties[active])

        my_bankroll = game_state.bankroll
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1 - active]
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        continue_cost = opp_pip - my_pip
        my_contrib = STARTING_STACK - my_stack
        opp_contrib = STARTING_STACK - opp_stack
        pot_size = my_contrib + opp_contrib
        min_raise, max_raise = round_state.raise_bounds()
        min_raise = min(min_raise + BIG_BLIND * random.randint(3, 6), max_raise) # 3, 18

        my_cards = [eval7.Card(card) for card in my_cards]
        board_cards = [eval7.Card(card) for card in board_cards]
        cards = my_cards + board_cards
        hole_type = eval7.handtype(eval7.evaluate(my_cards))
        board_type = eval7.handtype(eval7.evaluate(board_cards))
        hand_type = eval7.handtype(eval7.evaluate(cards))
        hand_suited = my_cards[0].suit == my_cards[1].suit
        hand_rating = rate_hand(my_cards)

        my_high_rank = 0
        for card in my_cards:
            if card.rank > my_high_rank:
                my_high_rank = card.rank


        # strategic fold and tank blinds
        if self.auto_fold:
            # prefer fold over check, prevent opp from seeing more cards
            return FoldAction()

        # check through if only option, i.e. all in
        if my_stack == 0 and CheckAction in legal_actions:
            return CheckAction()


        if street == 0:
            hand_percentile = hand_rating[1] / len(RATINGS)

            if hand_percentile > .52:
                return self.check_fold(legal_actions)
            elif hand_percentile < .25:
                return self.raise_by(1/8, round_state)
            elif hand_percentile < .4:
                return self.raise_by(min_raise, round_state)
            else:
                if continue_cost < random.randint(8, 18):
                    if CheckAction in legal_actions:
                        return CheckAction()
                    if CallAction in legal_actions:
                        return CallAction()

        else:
            ev = self.estimate_ev(my_cards, board_cards, my_bounty, pot_size)

            if hand_type != "High Card":
                if hand_type == board_type:
                    ev += 15
                    if hand_type == "Pair":
                        if ev > 0:
                            return self.raise_by(1/6, round_state)
                        else:
                            if CheckAction in legal_actions:
                                return CheckAction()
                            if continue_cost < abs(ev) / 2:
                                return CallAction()
                    elif hand_type == "Two Pair" or hand_type == "Trips":
                        if ev > 0:
                            return self.raise_by(1/5, round_state)
                        else:
                            if CheckAction in legal_actions:
                                return CheckAction()
                            if continue_cost < abs(ev) / 2:
                                return CallAction()
                    else:
                        # check/fold if low kicker
                        if hand_type == "Quads":
                            if my_high_rank <= 8:
                                return self.check_fold(legal_actions)

                        # attempt to bully, hoping they don't have chops implemented
                        # if they have a higher straight/flush/whatever, oh well

                        return self.raise_by(max_raise, round_state)
                else:
                    if hand_type == "Pair":
                        if ev > 0:
                            ev = min(ev, STARTING_STACK / 5)
                            return self.raise_by(ev, round_state)
                        else:
                            if continue_cost < abs(ev) / 2:
                                if my_pip == 0:
                                    if RaiseAction in legal_actions:
                                        return self.raise_by(min_raise, round_state)
                                if CheckAction in legal_actions:
                                    return CheckAction()
                                if CallAction in legal_actions:
                                    return CallAction()
                    else:
                        return self.raise_by(3.2/8, round_state)
            else:
                if continue_cost == 0:
                    if ev > -50:
                        if RaiseAction in legal_actions:
                            return self.raise_by(min_raise, round_state)

                if ev > 0:
                    ev = min(ev, STARTING_STACK / 5)
                    return self.raise_by(ev, round_state)
                else:
                    if continue_cost < abs(ev) / 2:
                        if my_pip == 0:
                            return self.raise_by(min_raise, round_state)
                        return CallAction()


        return self.check_fold(legal_actions)


    def estimate_ev(self, my_cards, board_cards, my_bounty, pot_size):
        t0 = time.time()
        trials = 5000 # 20000, 3000, 1500
        delta = 0
        streets_left = 5 - len(board_cards)
        bounty_size = math.ceil(pot_size * .5) + 10

        my_bounty_hit = False
        for card in my_cards + board_cards:
            if card.rank == my_bounty:
                my_bounty_hit = True
                break

        deck = eval7.Deck()
        deck.shuffle()
        for card in my_cards + board_cards:
            deck.cards.remove(card)


        for opp_bounty in self.opp_bounty_pos:
            opp_bounty_hit = False
            for card in board_cards:
                if card.rank == opp_bounty:
                    opp_bounty_hit = True
                    break

            for i in range(trials):
                cards = deck.sample(2 + streets_left)
                opp_cards = cards[:2]
                trial_cards = cards[2:]
                trial_board = board_cards + trial_cards
                my_eval = eval7.evaluate(my_cards + trial_board)
                opp_eval = eval7.evaluate(opp_cards + trial_board)

                if not my_bounty_hit:
                    for card in trial_cards:
                        if card.rank == my_bounty:
                            my_bounty_hit = True
                            break
                if not opp_bounty_hit:
                    for card in opp_cards + trial_cards:
                        if card.rank == opp_bounty:
                            opp_bounty_hit = True
                            break

                if my_eval > opp_eval:
                    delta += pot_size + my_bounty_hit and bounty_size or 0
                elif my_eval < opp_eval:
                    delta -= pot_size + opp_bounty_hit and bounty_size or 0
                else:
                    delta += my_bounty_hit and bounty_size or 0
                    delta -= opp_bounty_hit and bounty_size or 0

        self.time_thinking += time.time() - t0
        return delta / trials

    def check_fold(self, legal_actions):
        return CheckAction in legal_actions and CheckAction() or FoldAction()

    def raise_by(self, amount, round_state):
        legal_actions = round_state.legal_actions()
        min_raise, max_raise = round_state.raise_bounds()
        o_amount = amount

        if amount <= 1:
            # amount = (amount) * (max_raise - min_raise)
            amount = amount * STARTING_STACK + min_raise / 2
        amount = amount * (1 + (random.random() - .5) / 5)
        amount = math.floor(min(max(amount, min_raise), max_raise))

        if amount > STARTING_STACK / 2.6: # just all in
            amount = max_raise

        if RaiseAction in legal_actions:
            return RaiseAction(amount)
        if CallAction in legal_actions:
            return CallAction()
        return CheckAction()


if __name__ == "__main__":
    run_bot(Player(), parse_args())
