
import eval7
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import calculate_equity, huber_loss
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND

# Define the DQN model
class PokerDQN(nn.Module):
    def __init__(self, input_size,  output_size):
        # equity and counter class variables
        self.cache = {}
        self.round_counter = 0

        super(PokerDQN, self).__init__()
        # NN architechture
        layers = [nn.Linear(input_size, 128)]
        for i in range(3, 0, -1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(2**(i+4), 2**(i+3)))

        layers.append(nn.ReLU())
        layers.append(nn.Linear(2**4, output_size))

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        output = self.fc(x)
        return output

    # Function to preprocess game state and round state into a numerical input vector
    def preprocess_state(self, game_state, round_state, active):
        """
        Convert the game state and round state into a numerical feature vector. Feature vector helps the model
        make sense of what is going on in the game. Helpful inputs include hand equity, cards, and opponent actions.
        Arguments:
            GameState: game state class from engine
            RoundState: round state class from engine
            int: active, gives the seat of the bot in a given round (BB or SB)

        Returns:
            torch.Tensor: Preprocessed feature vector.
        """
        features = []

        # adding numerical representations of my and board cards
        street = round_state.street
        ranks = [(eval7.Card(card).rank / 14) for card in round_state.hands[active]]
        suits = [(eval7.Card(card).suit / 3) for card in round_state.hands[active]]

        board_ranks = [(eval7.Card(card).rank / 14) for card in round_state.deck[:street]]
        board_suits = [(eval7.Card(card).suit / 3) for card in round_state.deck[:street]]

        while len(board_ranks) < 5:
            board_ranks.append(0)
            board_suits.append(0)

        # adding my and opp pip, stacks, and contributions
        my_pip = round_state.pips[active]  # the number of chips you have contributed to the pot this round of betting
        opp_pip = round_state.pips[1-active]  # the number of chips your opponent has contributed to the pot this round of betting
        my_stack = round_state.stacks[active]  # the number of chips you have remaining
        opp_stack = round_state.stacks[1-active]  # the number of chips your opponent has remaining
        continue_cost = opp_pip - my_pip  # the number of chips needed to stay in the pot
        my_bounty = round_state.bounties[active]  # your current bounty rank
        my_contribution = STARTING_STACK - my_stack  # the number of chips you have contributed to the pot
        opp_contribution = STARTING_STACK - opp_stack  # the number of chips your opponent has contributed to the pot

        my_bounty = my_bounty + 's'
        my_bounty = eval7.Card(my_bounty).rank / 14

        features = ranks + suits + board_ranks + board_suits
        features.extend([my_pip, opp_pip, my_stack, opp_stack, continue_cost, my_bounty, my_contribution, opp_contribution])
        features.append(street)

        # adding bankroll, game_clock, big and small blinds
        bankroll = game_state.bankroll
        game_clock = game_state.game_clock

        self.round_counter += 1
        game_progress = self.round_counter / NUM_ROUNDS

        features.extend([bankroll, SMALL_BLIND, BIG_BLIND, game_progress, game_clock])

        # calculating hand equity and pot odds, adding to features
        self.cache, hand_equity = calculate_equity(self.cache, round_state.hands[active])
        if continue_cost > 0:
            pot_odds = (continue_cost) / (my_contribution + opp_contribution + continue_cost)
        else:
            pot_odds = 0
        features.extend([hand_equity, pot_odds])

        return torch.tensor(features, dtype=torch.float32)


# Function to predict the action using epsilon-greedy policy
def predict_action(model, game_state, round_state, active, epsilon = 0.01):
    '''
    Predict action using epsilon-greedy policy. Action space includes fold, check, call, and 4 types of raises.
    performs inference and predicts action with highest Q-value

    Arguments:
        PokerDQN: model instance
        GameState: game_state object from engine
        RoundState: round_state object from engine
        int: active player seat (BB or SB)
        float: epsilon gives frequency to perform a random action. Helps the model explore the action space.

    Return:
        tuple: (action (name of action), action_id (id in list of actions), raise_amount)
    '''
    # Preprocess the input state
    input_vector = model.preprocess_state(game_state, round_state, active)
    input_vector = input_vector.unsqueeze(0)

    # define action space
    min_raise, max_raise = round_state.raise_bounds()
    pot = 2*STARTING_STACK - round_state.stacks[active] - round_state.stacks[1-active]

    ACTIONS = ["FoldAction", "CheckAction", "CallAction"] + 5*["RaiseAction"]
    raise_amounts = [0, 0, 0, min_raise, min(0.5*pot, max_raise), min(pot, max_raise), min(2*pot, max_raise), max_raise]

    if random.random() < epsilon:
        #Random action
        action_id = random.randint(0, 7)
        return ACTIONS[action_id], action_id, raise_amounts[action_id]

    else:
        #Use the model to predict the best action
        output = model(input_vector)
        action_id = torch.argmax(output, dim=1).item()

        return ACTIONS[action_id], action_id, raise_amounts[action_id]


# Function to update the model using Q-learning
def update_model(model, target_model, optimizer, state_vector, action_idx, reward, next_state_vector, gamma, terminal):
    """
    Update the model using the Q-learning formula. Reward is the Q-value of the next state or the delta if the round ended.

    Args:
        PokerDQN: model is he current Q-network.
        PokerDQN: target_model is the target Q-network; stabalizes training
        Optimizer: Optimizer for the model.
        torch.Tensor: state_vector is the current state as a preprocessed input vector.
        int: action_id is the index of the action taken.
        int: reward (delta) received from the action, if any.
        torch.Tensor: next_state_vector as a preprocessed input vector.
        float: Gamma is a discount factor for future rewards.
        bool: Terminal true if round ended.
    """
    # Add batch dimension to state vectors
    state_vector = state_vector.unsqueeze(0)
    next_state_vector = next_state_vector.unsqueeze(0)

    # Compute Q-value for the taken action
    q_values = model(state_vector)
    q_value = q_values[0, action_idx]

    # Compute target Q-value
    with torch.no_grad():
        if terminal:
            target_q_value = torch.tensor(reward, dtype=q_value.dtype)  # Use reward directly for terminal state
        else:
            next_q_values = target_model(next_state_vector)
            max_next_q_value = torch.max(next_q_values)
            target_q_value = reward + gamma * max_next_q_value

    # Compute loss
    loss = huber_loss(q_value, target_q_value, 100) # Huber rather than MSE to not overfit outlier hands

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Updated model with reward {reward:.2f}, target Q-value {target_q_value:.2f}, terminal={terminal}")
