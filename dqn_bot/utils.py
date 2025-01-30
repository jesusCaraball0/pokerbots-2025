import eval7
import torch


def calculate_equity(cache, hand, num_trials=1000):
    '''
    Uses Monte Carlo simulations of poker rounds to estimate the equity of a given hand

    Arguments:
        Dictionary: cache contains the equity of a hand if it has previously been calculated
        List: hand is a list of 2 eval7.Card() objects holding the player's current hand
        int: num_trials that the Monte Carlo simulation is run
    Returns:
        float: decimal representation of the given hand's equity
    '''
    if str(hand) in cache:
        return cache, cache[str(hand)]

    deck = eval7.Deck()
    for card in hand:
        deck.cards.remove(eval7.Card(card))

    wins = 0
    for _ in range(num_trials):
        deck.shuffle()

        community_cards = deck.peek(5)
        opponents = deck.peek(2)

        my_hand = [eval7.Card(c) for c in hand] + community_cards
        my_score = eval7.evaluate(my_hand)

        opponent_hand = opponents + community_cards
        opponent_score = eval7.evaluate(opponent_hand)

        if my_score > opponent_score:
            wins += 1
        elif my_score == opponent_score:
            wins += 0.5

    cache[str(hand)] = (wins / num_trials)
    return cache, (wins / num_trials)

def save_model(model, filename):
    '''
    saves the model at the end of each game
    Arguments:
        PokerDQN: active model
        string: filename to save to
    '''
    torch.save(model.state_dict(), filename)
    print(f"model saved to {filename}")

def huber_loss(q_value, target_q_value, delta=200):
    '''
    computes the loss based on the Huber function. Minimizes overfitting to outlier hands.
    Arguments:
        torch.tensor: q_value is the model's current estimate of the action's EV
        torch.tensor: target_q_value is the target model's or actual reward for the action
        int: delta is the treshold value to switch from quadratic to linear loss.
    Returs:
        torch.float32: loss for the given parameters.
    '''
    diff = torch.abs(q_value - target_q_value)
    loss = torch.where(diff <= delta, 0.5*diff**2, delta*(diff-0.5*delta))
    return loss.mean()
