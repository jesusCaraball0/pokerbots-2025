
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
        # equity class variable
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
        Convert the game state and round state into a numerical feature vector.
        This function must be customized based on the specific poker game and data format.

        Example:
            - Player's cards (encoded as one-hot vectors or numerical indices)
            - Board cards (encoded similarly)
            - Stack sizes (normalized values)
            - Opponent action history (e.g., last action, bet amount)

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
    takes in model and current state and performs inference to find optimal action
    return tuple: (Action, action_id, raise_amount)
    '''
    # Preprocess the input state
    input_vector = model.preprocess_state(game_state, round_state, active)

    # Add batch dimension (for a single example)
    input_vector = input_vector.unsqueeze(0)

    # define action space
    min_raise, max_raise = round_state.raise_bounds()
    pot = 2*STARTING_STACK - round_state.stacks[active] - round_state.stacks[1-active]

    ACTIONS = ["FoldAction", "CheckAction", "CallAction"] + 5*["RaiseAction"]
    raise_amounts = [0, 0, 0, min_raise, min(0.5*pot, max_raise), min(pot, max_raise), min(2*pot, max_raise), max_raise]

    if random.random() < epsilon:
        # Explore: Random action
        action_id = random.randint(0, 7)
        return ACTIONS[action_id], action_id, raise_amounts[action_id]

    else:
        # Exploit: Use the model to predict the best action
        output = model(input_vector)
        action_id = torch.argmax(output, dim=1).item()

        return ACTIONS[action_id], action_id, raise_amounts[action_id]


# Function to update the model using Q-learning
def update_model(model, target_model, optimizer, state_vector, action_idx, reward, next_state_vector, gamma, terminal):
    """
    Update the model using the Q-learning formula, accounting for terminal and non-terminal states.

    Args:
        model (PokerDQN): The current Q-network.
        target_model (PokerDQN): The target Q-network.
        optimizer (Optimizer): Optimizer for the model.
        state_vector (torch.Tensor): Current state as a preprocessed input vector.
        action_idx (int): Index of the action taken.
        reward (float): Reward (delta) received from the action, if any.
        next_state_vector (torch.Tensor): Next state as a preprocessed input vector.
        gamma (float): Discount factor for future rewards.
        terminal (bool): Whether the current state is a terminal state.
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
    loss = huber_loss(q_value, target_q_value, 100) # Huber rather than MSE to handle outlier hands

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #print(f"Updated model with reward {reward:.2f}, target Q-value {target_q_value:.2f}, terminal={terminal}")
