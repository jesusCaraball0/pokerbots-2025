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


cache = set()
starting_hands = []
for rank1 in eval7.ranks:
    for rank2 in eval7.ranks:
        hand_key = eval7.evaluate([eval7.Card(card) for card in [rank1 + "s", rank2 + "c"]])
        if hand_key not in cache:
            cache.add(hand_key)
            starting_hands.append(rank1 + rank2)
range_all = eval7.HandRange(",".join(starting_hands))


class Player(Bot):
    '''
    chirp
    '''


    def __init__(self):
        # stats
        self.fold_counter = 0
        self.chop_counter = 0
        self.call_win_counter = {}
        self.time_thinking = 0
        self.num_evals = 0

        # game states
        self.auto_fold = False
        self.opp_projected_win = False
        self.opp_call_win_counter = {}

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
        # total_prob = 0
        # for pos in self.opp_bounty_pos:
        #     i=0#amount of opp bounty cards in my hand
        #     for card in my_cards:
        #         if card.rank==pos:
        #             i+=1
        #     prob=(4-i)/50+(50-4+i)*(4-i)/(50*49)
        #     total_prob+=prob/len(self.opp_bounty_pos)
        # bounty_rate=total_prob * 1.1
        # above is actually not working as expected
        # forgot that bounties change
        # assume i = 0, i.e. 4/50 + 46/50  * 4/49

        bounty_rate = (4/50 + 46/50 * 4/49) * 1.15
        max_payment = blind_cost + math.ceil(bounty_cost)
        remaining_payment = blind_cost + math.ceil(bounty_cost * bounty_rate) # fold if above this threshold

        if my_bankroll - remaining_payment > 20:
            if not self.auto_fold:
                self.auto_fold = True
                # print(bounty_rate)
                print(f"STRATEGIC FOLD @ {round_num}, ${my_bankroll}\t\t(${remaining_payment}, MAX ${max_payment})")


        # opp will probably win, play more risky before they start tanking blinds
        if not self.auto_fold:
            if opp_bankroll > blind_cost + bounty_rate * bounty_cost: # .8
                if not self.opp_projected_win:
                    print("opp projected to win, be more agressive", round_num)
                    self.opp_projected_win = True
            elif self.opp_projected_win:
                if opp_bankroll < blind_cost * .2:
                    print("INVERTED LOSS, opp was originally projected to win", round_num)
                    self.opp_projected_win = False


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
        time_used = 60 - game_state.game_clock


        # discern opponent bounty
        if round_num % 25 == 0: # bounties reset every 25 rounds
            self.opp_bounty = None
            self.opp_bounty_pos = set(range(0, 13))
        elif len(self.opp_bounty_pos) > 1: # bounty has yet to be determined
            pos = set([eval7.Card(card).rank for card in opp_cards + board_cards])
            if opponent_bounty_hit:
                if opp_cards:
                    self.opp_bounty_pos.intersection_update(pos)
            elif my_delta <= 0:
                self.opp_bounty_pos.difference_update(pos)
            if len(self.opp_bounty_pos) == 1: # bounty determined
                self.opp_bounty = next(iter(self.opp_bounty_pos))


        # record stats
        if my_delta == 0:
            self.chop_counter += 1

        if my_contrib > (bool(active) and BIG_BLIND or SMALL_BLIND):
            if my_delta > 0:
                self.call_win_counter[round_num] = 1
            else:
                self.call_win_counter[round_num] = 0

        if opp_contrib > (bool(1 - active) and BIG_BLIND or SMALL_BLIND):
            if my_delta > 0:
                self.opp_call_win_counter[round_num] = 1
            else:
                self.opp_call_win_counter[round_num] = 0

        if my_contrib < opp_contrib and not self.auto_fold:
            self.fold_counter += 1

        # print(f"{round_num}, {game_state.bankroll}, {-game_state.bankroll}")


        # match over
        if round_num == NUM_ROUNDS:
            call_win_counter = sum(self.call_win_counter.values())
            num_calls = len(self.call_win_counter)
            opp_call_win_counter = sum(self.opp_call_win_counter.values())
            opp_num_calls = len(self.opp_call_win_counter)

            print("\nStats:")
            print("Fold %:", self.fold_counter / NUM_ROUNDS * 100, self.fold_counter)
            print("Chop %:", self.chop_counter / NUM_ROUNDS * 100, self.chop_counter)
            print("Call Win %:", call_win_counter / num_calls * 100, call_win_counter, num_calls)
            print("Opp Call Win %:", opp_call_win_counter / opp_num_calls * 100, opp_call_win_counter, opp_num_calls)
            print("Time Spent Thinking (s):", round(self.time_thinking, 2))
            print("Avg Time Spent Thinking (ms):", round(self.time_thinking / max(self.num_evals, 1) * 1000, 2) , f"(x{self.num_evals})")
            print("Total Time Used (s):", round(time_used, 2), f"(d={round(time_used - self.time_thinking, 2)})")
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
        min_raise = min(min_raise + BIG_BLIND * random.randint(2, 4), max_raise) # 3, 18
        pot_odds = continue_cost / pot_size

        my_cards = [eval7.Card(card) for card in my_cards]
        board_cards = [eval7.Card(card) for card in board_cards]
        cards = my_cards + board_cards
        hole_type = eval7.handtype(eval7.evaluate(my_cards))
        board_eval = eval7.evaluate(board_cards)
        hand_eval = eval7.evaluate(cards)
        board_type = eval7.handtype(board_eval)
        hand_type = eval7.handtype(hand_eval)
        hand_suited = my_cards[0].suit == my_cards[1].suit
        hand_rating = rate_hand(my_cards)

        my_high_rank = 0
        for card in my_cards:
            if card.rank > my_high_rank:
                my_high_rank = card.rank


        # strategic fold and tank blinds
        # self.auto_fold = False
        if self.auto_fold:
            # prefer fold over check, prevent opp from seeing more cards
            if hand_rating[0] * 3 < 60: # TODO: tune experimental values
                return FoldAction()

        # check through if only option, i.e. all in
        if my_stack == 0 and CheckAction in legal_actions:
            return CheckAction()

        if self.opp_projected_win:
            if CheckAction in legal_actions:
                return CheckAction()
            if continue_cost <= BIG_BLIND * 2:
                return self.raise_by(min_raise, round_state)

        ev = equity = ev_raise = 0
        if street == 0:
            ev = hand_rating[0] * 3
        else:
            ev, equity, ev_raise = self.estimate_ev(my_cards, range_all, board_cards, my_bounty, pot_size, continue_cost)


        # print(round_num, my_cards, board_cards, ev, my_bankroll)
        if street == 0:
            if ev < 0:
                # TODO: play tighter with negative ev (i.e. maybe fold more?)
                if continue_cost > BIG_BLIND * 3 and continue_cost > abs(ev):
                    return self.check_fold(legal_actions)
                else:
                    if CallAction in legal_actions:
                        return CallAction()

            if my_contrib + continue_cost < my_bankroll / 100: # TODO: tune experimental value
                if CallAction in legal_actions:
                    return CallAction()
            if my_pip == 0 and continue_cost < ev:
                return self.raise_by(min_raise + continue_cost, round_state)
            if continue_cost < ev and continue_cost < 20:
                if CallAction in legal_actions:
                    return CallAction()
        else:
            if hand_type != "High Card":
                if hand_type == board_type:
                    # we are essentially playing the board, besides some edge cases
                    # in these edge cases, if we have the near-nuts, we will tend to play agressively, which may not be optimal, but it prevents us from getting bullied
                    # otherwise, play as reserved yet agressive as possible

                    proxy_board_cards = []
                    proxy_board_ranks = set()
                    for card in board_cards:
                        if eval7.handtype(eval7.evaluate([p_card for p_card in board_cards if p_card != card])) != hand_type:
                            proxy_board_cards.append(card)
                            proxy_board_ranks.add(card.rank)

                    proxy_cards = []
                    proxy_ranks = set()
                    for card in cards:
                        for rank in proxy_board_ranks:
                            if eval7.handtype(eval7.evaluate([p_card for p_card in cards if p_card != card and p_card.rank != rank])) == hand_type:
                                proxy_cards.append(card)
                                proxy_ranks.add(card.rank)
                                break
                    proxy_ranks = proxy_ranks.union(proxy_board_ranks)

                    my_rank = max(proxy_ranks)
                    board_rank = max(proxy_board_ranks)
                    if my_rank > board_rank:
                        print(round_num, "HIGHER SET", hand_type, my_cards, board_cards)


                    adj_equity = self.adjusted_equity(my_cards, board_cards, equity, hand_type, pot_odds) # TODO: needs to be tuned!!!!!!!!
                    # TODO: go all in if board is already the (near) nuts, hope opp doesn't notice

                    if hand_type == "Pair":
                        if random.random() < adj_equity * 2: # TODO: TUNE THE FUNC!!! :(
                            # print("\t\t\t\ttest", hand_type, adj_equity, my_cards, board_cards, min_raise + continue_cost)
                            if continue_cost == 0:
                                return self.raise_by(min_raise + continue_cost, round_state)
                            else:
                                # hurts my head, need to do the logic to make sure we arent calling all-ins all the time
                                # but that might already be included in the previous random.random(), in which case it doesn't really matter
                                # god idfk, this is good enough for now though
                                if CallAction in legal_actions:
                                    return CallAction()
                    elif hand_type == "Two Pair" or hand_type == "Trips":
                        # opp would need make higher two pair or a boats
                        if hand_type == "Two Pair" and my_rank > board_rank:
                            return self.raise_by(min_raise + continue_cost, round_state)

                        # opp would need to make higher trips or a boat
                        if hand_type == "Trips" and my_rank > board_rank:
                            return self.raise_by((min_raise + continue_cost) * 2, round_state)

                        print(round_num, hand_type, equity, adj_equity, pot_odds, my_cards, board_cards)
                        pass
                    elif hand_type == "Full House" or hand_type == "Quads":
                        # opp would need a higher pocket pair
                        if hand_type == "Full House" and my_rank > board_rank:
                            return self.raise_by((min_raise + continue_cost) * 2, round_state)

                        pass
                    elif hand_type == "Straight" or hand_type == "Flush":
                        # opp would need back-to-back connectors
                        if hand_type == "Straight" and my_rank > board_rank:
                            return self.raise_by((min_raise + continue_cost) * 2, round_state)

                        # TODO: check for num of higher flushes, and eval the chance they have the higher suits
                        # TODO: if its a mid straight (specifically not a high straight), we are probably going to chop
                    else:
                        # i mean, very small chance to have a higher straight flush, the num of max outs is literally 1
                        # hope opp doesn't realize
                        print("ANOMALY !!!!!", round_num, hand_type, my_cards, board_cards, my_stack, opp_stack)
                        return self.raise_by(max_raise, round_state)
                else:
                    if hand_type == "Pair":
                        pass
                    else:
                        pass

            # the old base logic, we'll use it as a fallback for any yet-undefined behavior

            if ev <= 0:
                if random.random() > equity * .8:
                    return self.check_fold(legal_actions)
                else:
                    # bluff ?
                    ev += BIG_BLIND * 3
                    equity += .3
            if continue_cost == 0 and equity > .65 and random.random() > .35:
                return self.raise_by(min_raise + continue_cost, round_state)
            if continue_cost < ev and equity > .5:
                if CheckAction in legal_actions:
                    return CheckAction()
                if CallAction in legal_actions:
                    return CallAction()
            if continue_cost < ev * 1.2: # 1.2, .7, 1
                if CallAction in legal_actions:
                    if random.random() < equity * 1.5:
                            return CallAction()
                # equity = self.adjusted_equity(my_cards, board_cards, pot_odds)
                # if equity > pot_odds:
                #     if CallAction in legal_actions:
                #         return CallAction()

        return self.check_fold(legal_actions)


    def get_ev_raise(self, equity, pot_size, continue_cost, my_bounty_prob, opp_bounty_prob):
        # pot_size += continue_cost
        # bounty_size = math.ceil(pot_size * .5) + 10
        # ev_raise = equity == 1 and math.inf or ((equity * pot_size
        #             + equity * my_bounty_prob * bounty_size
        #             - (1 - equity) * opp_bounty_prob * bounty_size)
        #             / (1 - equity))
        ev_raise = pot_size / (1 / max(min(equity, .5), 0) - 1)
        # print("EV RAISE:", ev_raise, my_bounty_prob, opp_bounty_prob)
        return ev_raise

    def adjusted_equity(self, my_cards, board_cards, equity, hand_type, pot_odds):
        t0 = time.time()
        # pot_odds = max(pot_odds, .5) / 1.5
        # num = 0
        # avg = 0

        # for hand in starting_hands:
        #     equity = eval7.py_hand_vs_range_monte_carlo(my_cards, eval7.HandRange(hand), board_cards, 1500)
        #     if (1 - equity) > pot_odds:
        #         num += 1
        #         avg += equity
        # return num != 0 and avg / num or 1

        # pairs = eval7.HandRange("AA, KK, QQ, JJ, TT, 99, 88, 77, 66, 55, 44, 33, 22")
        # equity = eval7.py_hand_vs_range_monte_carlo(my_cards, pairs, board_cards, 3500)

        equity = equity - pot_odds

        self.time_thinking += time.time() - t0
        self.num_evals += 1
        return equity

    def estimate_ev(self, my_cards, opp_range, board_cards, my_bounty, pot_size, continue_cost):
        t0 = time.time()
        streets_left = 5 - len(board_cards)
        equity = (eval7.py_hand_vs_range_exact(my_cards, opp_range, board_cards) if streets_left == 0 else
                eval7.py_hand_vs_range_monte_carlo(my_cards, opp_range, board_cards, 3500)) # 1000000, 500000, ~150000, 50000, 3500
        bounty_size = math.ceil(pot_size * .5) + 10

        deck = eval7.Deck()
        for card in my_cards + board_cards:
            deck.cards.remove(card)
        cards_left = len(deck.cards) # - 2
        board_ranks = set(card.rank for card in board_cards)

        my_bounty_prob = any(card.rank == my_bounty for card in my_cards + board_cards) and 1 or 0
        if my_bounty_prob == 0:
            # calculate worst-case-scenario/adverserial chance that NONE of the future streets strike our bounty
            no_street_bounty_prob = 1
            for i in range(streets_left):
                no_street_bounty_prob *= (cards_left - 4 - i) / (cards_left - i)

            my_bounty_prob = 1 - no_street_bounty_prob

        opp_bounty_prob = 0
        for opp_bounty in self.opp_bounty_pos:
            if opp_bounty in board_ranks: # if the bounty is already on the board
                opp_bounty_prob += 1
            else:
                # else, how many are left in the deck (i.e. we know they were dealt as our hole cards)
                bounty_freq = sum(1 for card in deck.cards if card.rank == opp_bounty)

                # calculate worst-case-scenario/adverserial chance that opp DOES HAVE (having is NOT the same as drawing) their bounty
                # opp_doesnt_have_their_bounty = (cards_left - bounty_freq)/cards_left * (cards_left - 1 - bounty_freq)/(cards_left - 1)
                opp_has_their_bounty = bounty_freq/cards_left + (cards_left - bounty_freq)/cards_left * bounty_freq/(cards_left - 1)

                # calculate worst-case-scenario/adverserial chance that at least one of the future streets strike their bounty
                street_bounty_prob = streets_left != 0 and bounty_freq/cards_left or 0
                for i in range(1, streets_left):
                    street_bounty_prob += bounty_freq/(cards_left - i) * (cards_left - i - 1 - bounty_freq)/(cards_left - i - 1)

                # calculate worst-case-scenario/adverserial chance that NONE of the future streets strike their bounty
                no_street_bounty_prob = 1
                for i in range(streets_left):
                    no_street_bounty_prob *= (cards_left - bounty_freq - i) / (cards_left - i)

                # average b/c I can't get the statistics quite right on this one (??)
                # off by very little regardless
                street_bounty_prob = (street_bounty_prob + 1 - no_street_bounty_prob) / 2

                opp_bounty_prob += opp_has_their_bounty + street_bounty_prob * (1 - opp_has_their_bounty)
        opp_bounty_prob /= len(self.opp_bounty_pos)

        # print(equity, pot_size, continue_cost, str(round((time.time() - t0) * 1000, 2)) + " ms")
        self.time_thinking += time.time() - t0
        self.num_evals += 1
        return ((equity * pot_size
                + equity * my_bounty_prob * bounty_size
                - (equity * (1 - equity)) * (my_bounty_prob * (1 - opp_bounty_prob)) * bounty_size / 2
                - (1 - equity) * continue_cost
                - (1 - equity) * opp_bounty_prob * bounty_size
                + ((1 - equity) * equity) * (opp_bounty_prob * (1 - my_bounty_prob)) * bounty_size / 2),
                equity,
                self.get_ev_raise(equity, pot_size, continue_cost, my_bounty_prob, opp_bounty_prob))

    def check_fold(self, legal_actions):
        return CheckAction() if CheckAction in legal_actions else FoldAction()

    def raise_by(self, amount, round_state):
        # TODO: figure out standard practice for our team for what we mean by raising i.e. raising to an amount vs raising by
        legal_actions = round_state.legal_actions()
        min_raise, max_raise = round_state.raise_bounds()
        o_amount = amount

        if amount <= 1:
            # amount = (amount) * (max_raise - min_raise)
            amount = amount * STARTING_STACK + min_raise / 2
        amount = amount * (1 + (random.random() - .5) / 5 * 2) # +/- 20%
        amount = math.floor(min(max(amount, min_raise), max_raise))

        if amount > STARTING_STACK / 2.6: # just all in
            amount = max_raise

        if RaiseAction in legal_actions:
            return RaiseAction(amount)
        if CallAction in legal_actions:
            return CallAction()
        if CheckAction  in legal_actions:
            return CheckAction()
        return FoldAction()


if __name__ == "__main__":
    run_bot(Player(), parse_args())
