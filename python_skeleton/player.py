'''
Simple example pokerbot, written in Python.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
import eval7
import time
import random
import math


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
        self.bounty_possibilities={0,1,2,3,4,5,6,7,8,9,10,11,12}#for opponent
        self.time_thinking = 0
        self.auto_fold = False

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

        bounty_rate = 4/50 + 46/50 * 4/49
        max_payment = blind_cost + math.ceil(bounty_cost)
        remaining_payment = blind_cost + math.ceil(bounty_cost * bounty_rate) # fold if above this threshold

        if my_bankroll - remaining_payment > 20:
            if not self.auto_fold:
                self.auto_fold = True
                print(f"STRATEGIC FOLD @ {round_num}, ${my_bankroll}\t\t(${remaining_payment}, MAX ${max_payment})")

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
        opp_cards = previous_state.hands[1-active]  # opponent's cards or [] if not revealed
        round_num=game_state.round_num
        my_bounty_hit = terminal_state.bounty_hits[active]  # True if you hit bounty
        opponent_bounty_hit = terminal_state.bounty_hits[1-active] # True if opponent hit bounty
        bounty_rank = previous_state.bounties[active]  # your bounty rank
        street=previous_state.street
        board_cards = previous_state.deck[:street]
        if(opponent_bounty_hit and opp_cards):
            pos=set()
            for card in opp_cards+board_cards:
                pos.add(eval7.Card(card).rank)


            self.bounty_possibilities=self.bounty_possibilities.intersection(pos)
        if(round_num%25==0):
            self.bounty_possibilities=set(range(0,13))

        if round_num == NUM_ROUNDS:
            print("Time Spent Thinking (s):", round(self.time_thinking, 2))

        # The following is a demonstration of accessing illegal information (will not work)
        #opponent_bounty_rank = previous_state.bounties[1-active]  # attempting to grab opponent's bounty rank

        #if my_bounty_hit:
            #print("I hit my bounty of " + bounty_rank + "!")
        #if opponent_bounty_hit:
            #print("Opponent hit their bounty of " + opponent_bounty_rank + "!")
    def monte_carlo(self, my_cards, board_cards, my_bounty, pot_size, continue_cost):
        init_time=time.time()
        winnings=0#based off of continued cost as the only cost, ignoring sunk cost
        num_trials=700
        my_cards=[eval7.Card(my_cards[0]),eval7.Card(my_cards[1])]
        board_cards=[eval7.Card(board) for board in board_cards]
        continue_cost_mult=0#amount of times to subtract continue_cost from winnings
        #while(time.time()-init_time<55):
        deck=eval7.Deck()
        for card in my_cards+board_cards:
                deck.cards.remove(card)
        for i in range(num_trials):


            #deck.remove(my_cards)
            #Next three lines determine opponent's bounty
            # deck_opp_bounty=eval7.Deck()
            # deck_opp_bounty.shuffle()
            # i=0
            # opp_bounty=0
            # while(i<52-i and (deck_opp_bounty.peek(52-i)[i]).rank in self.bounty_possibilities):
            #     deck_opp_bounty.cards.remove((deck_opp_bounty.peek(52-i)[i]))
            #     i+=1
            # if i==52-i:
            #     opp_bounty=0
            # else:
            #     opp_bounty=deck_opp_bounty.peek(52-i)[i]
            opp_bounty=random.choice(list(self.bounty_possibilities))
            # if opp_bounty not in self.bounty_possibilities:
            #     opp_bounty=-1
            sampled_cards = deck.sample(2+5-len(board_cards))
            opp_cards=sampled_cards[:2]
            full_board=board_cards+sampled_cards[2:]
            self_val=eval7.evaluate(full_board+my_cards)
            opp_val=eval7.evaluate(full_board+opp_cards)
            #print("o",self_val, opp_val)
            if self_val>opp_val:
                bounty=False
                for card in full_board+my_cards:
                    #print(card, card.rank)
                    if card.rank==my_bounty:
                        winnings+=((1.5*pot_size+10)-continue_cost)
                        #winnings+=1.5*pot_size+10
                        #continue_cost_mult+=1
                        bounty=True
                        break
                if not bounty:
                    winnings+=pot_size-continue_cost
                    #continue_cost_mult+=1
            elif self_val<opp_val:
                bounty=False
                for card in full_board+opp_cards:
                    if card.rank==opp_bounty:
                        winnings-=((0.5*pot_size+10)+continue_cost)
                        #winnings-=((0.5*pot_size+10))
                        bounty=True
                        #continue_cost_mult+=1
                        break
                if not bounty:
                    winnings-=continue_cost
                    #continue_cost_mult+=1
            else:
                pass
                winnings+=.5*pot_size
                #winnings+=.5*pot_size+continue_cost #to account for subtracting continue cost later
        #print(f"{winnings=}")
        self.time_thinking += time.time() - init_time
        return winnings/num_trials




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

        # self.auto_fold = False
        if self.auto_fold:
            # prefer fold over check, prevent opp from seeing more cards
            return FoldAction()

        if RaiseAction in legal_actions:
           min_raise, max_raise = round_state.raise_bounds()  # the smallest and largest numbers of chips for a legal bet/raise
           min_cost = min_raise - my_pip  # the cost of a minimum bet/raise
           max_cost = max_raise - my_pip  # the cost of a maximum bet/raise
        exp_winnings=self.monte_carlo(my_cards, board_cards, my_cards, opp_contribution*2, continue_cost)
        if exp_winnings>continue_cost:
            if RaiseAction in legal_actions:
                return RaiseAction(min_raise+int((max_raise-min_raise)*random.random()/10))
            elif CallAction in legal_actions:
                return CallAction()
            return CheckAction()
        else:
            if random.random()<.1 or ((not board_cards) and random.random()<.80):
                if RaiseAction in legal_actions and random.random()<.05:
                    return RaiseAction(min_raise+int((max_raise-min_raise)*random.random()/12))
                elif CallAction in legal_actions:
                    return CallAction()

            elif CheckAction in legal_actions:
                return CheckAction()
            return FoldAction()
        # if not board_cards:
        #     my_cards=[eval7.Card(my_cards[0]),eval7.Card(my_cards[1])]
        #     board_cards=[eval7.Card(board) for board in board_cards]
        #     if (my_cards[0].rank>8 and my_cards[1].rank>8) or (my_cards[0].rank==my_cards[1].rank and my_cards[0].rank>=2):
        #         if random.random()>.90:
        #             if RaiseAction in legal_actions:
        #                 return RaiseAction(min_raise+int(random.random()*(max_raise-min_raise)))
        #             return CallAction()
        #         elif random.random()>.5 and CallAction in legal_actions:
        #             return CallAction()
        #         elif CheckAction in legal_actions:
        #             return CheckAction()
        #         else:
        #             return FoldAction()
        #     else:
        #         if CheckAction in legal_actions:
        #             return CheckAction()
        #         elif random.random()<.08:
        #             if CallAction in legal_actions:
        #                 return CallAction()
        #             if RaiseAction in legal_actions:
        #                 return RaiseAction(min_raise+int(random.random()*(max_raise-min_raise)/50))
        #         return FoldAction()
        # if RaiseAction not in legal_actions:
        #     if CheckAction in legal_actions:
        #         return CheckAction()
        #     exp_winnings=self.monte_carlo(my_cards,board_cards, my_bounty, 2*(opp_contribution),continue_cost)
        #     if (exp_winnings>continue_cost and random.random()<.95) or random.random()<.05:

        #         if CallAction in legal_actions:
        #             return CallAction()

        # raise_amt=random.random()*(max_raise-min_raise)+min_raise
        # exp_winnings=self.monte_carlo(my_cards,board_cards, my_bounty, 2*(opp_contribution+raise_amt),continue_cost+raise_amt)
        # print(f"{exp_winnings=}{my_cards=}{board_cards=}")
        # # with open("output.txt","w") as file:
        # #     file.write(exp_winnings)
        # #print(exp_winnings)
        # #print(exp_winnings, opp_contribution)
        # if (exp_winnings>continue_cost+raise_amt):# and random.random()<.95) or random.random()<.05:
        #     if RaiseAction in legal_actions:
        #         return RaiseAction(raise_amt)
        # else:
        #     if CheckAction in legal_actions:
        #         return CheckAction()
        #     exp_winnings=self.monte_carlo(my_cards,board_cards, my_bounty, 2*(opp_contribution),continue_cost)
        #     if (exp_winnings>continue_cost):# and random.random()<.95) or random.random()<.05:

        #         if CallAction in legal_actions:
        #             return CallAction()
        # if CheckAction in legal_actions:
        #     return CheckAction()

        # return FoldAction()
        # if RaiseAction in legal_actions:
        #     exp_winnings=self.monte_carlo(my_cards,board_cards, my_bounty)
        #     # with open("output.txt","w") as file:
        #     #     file.write(exp_winnings)
        #     if exp_winnings>0 and random.random()<exp_winnings*2:
        #         return RaiseAction(min_raise)
        # if CheckAction in legal_actions:  # check-call
        #     return CheckAction()
        # return FoldAction()
        # if random.random() < 0.25:
        #     return FoldAction()
        # return CallAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
