import eval7
import torch


def calculate_equity(cache, hand, num_trials=1000):
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
    torch.save(model.state_dict(), filename)
    print(f"model saved to {filename}")

def huber_loss(q_value, target_q_value, delta=200):
    diff = torch.abs(q_value - target_q_value)
    loss = torch.where(diff <= delta, 0.5*diff**2, delta*(diff-0.5*delta))
    return loss.mean()
