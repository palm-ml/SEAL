import torch

def generate_instance_independent_candidate_labels(train_labels, partial_rate):
    K = (torch.max(train_labels) - torch.min(train_labels) + 1).item()
    n = train_labels.shape[0]

    # Bernoulli Distribution
    Bernoulli_Matrix = torch.ones(n, K) * partial_rate
    Bernoulli_Matrix[torch.arange(n), train_labels] = 0
    incorrect_labels = torch.zeros(n, K)
    for i in range(n):
        incorrect_labels_sampler = torch.distributions.Bernoulli(probs=Bernoulli_Matrix[i])
        incorrect_labels_row = incorrect_labels_sampler.sample()
        while incorrect_labels_row.sum() < 1:
            incorrect_labels_row = incorrect_labels_sampler.sample()
        incorrect_labels[i] = incorrect_labels_row.clone().detach()
    # check
    partial_labels = incorrect_labels.clone().detach()
    partial_labels[torch.arange(n), train_labels] = 1.0
    avgC = partial_labels.sum() / n
    return partial_labels, avgC