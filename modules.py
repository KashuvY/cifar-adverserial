import torch

def return_probabilities(logits):
    probabilities = torch.nn.functional.softmax(logits, dim=1)  # calculate predicted probability distribution
    _, predicted_class = torch.max(logits, 1)  # returns (value, index), we are only insterested in the index
    return probabilities, predicted_class