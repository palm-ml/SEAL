import torch


class MetaLossNetwork(torch.nn.Module):

    def __init__(self, input_dim, output_dim, gamma, beta, reduction="mean",
                 logits_to_prob=True, one_hot_encode=True,
                 output_activation=torch.nn.Softplus(), **kwargs):

        """
        Creating a feed forward neural network meta loss function. Code inspired
        by the paper "Meta learning via learned loss." by Bechtle et al.

        :param output_dim: Output vector dimension of base network.
        :param reduction: Reduction operator for aggregating results.
        :param logits_to_prob: Apply transform to convert predicted output to probability.
        :param one_hot_encode: Apply transform to convert label to one-hot encoded label.
        :param output_activation: Loss function output activation.
        """

        super(MetaLossNetwork, self).__init__()

        # Meta-loss functions hyper-parameters.
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.reduction = reduction

        self.gamma = gamma
        self.beta = beta

        # Transformations to apply to the inputs.
        self.logits_to_prob = logits_to_prob
        self.one_hot_encode = one_hot_encode

        # Defining the loss functions architecture.
        self.network = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, 50, bias=False),
            SmoothLeakyReLU(gamma=self.gamma, beta=self.beta),
            torch.nn.Linear(50, 50, bias=False),
            SmoothLeakyReLU(gamma=self.gamma, beta=self.beta),
            torch.nn.Linear(50, 1, bias=False),
            output_activation
        )

        # Initializing the weights of the network.
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(module.weight)

        # for module in self.modules():
        #     if isinstance(module, nn.Linear):
        #         nn.init.xavier_uniform_(module.weight, gain=1.0)
        #         if module.bias is not None:
        #             module.bias.data.zero_()

    def forward(self, y_pred, y_target, part_y):
        n, c = y_pred.shape
        loss = self.network(
            torch.cat([
                y_pred.T.flatten().view(1, -1).T, 
                y_target.T.flatten().view(1, -1).T,
                part_y.T.flatten().view(1, -1).T,
                ], dim=1)
        ).view(c, n).sum(dim=0).mean()
        return loss

    def _transform_input(self, y_pred, y_target):

        if self.logits_to_prob:  # Converting the raw logits into probabilities.
            y_pred = torch.nn.functional.sigmoid(y_pred) if self.output_dim == 1 \
                else torch.nn.functional.softmax(y_pred, dim=1)

        if self.one_hot_encode:  # If the target is not already one-hot encoded.
            y_target = torch.nn.functional.one_hot(y_target, num_classes=self.output_dim)

        return y_pred, y_target

    def _reduce_output(self, loss):
        # Applying the desired reduction operation to the loss vector.
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
        
class SmoothLeakyReLU(torch.nn.Module):
    def __init__(self, gamma, beta) -> None:
        super().__init__()
        self.gamma = gamma
        self.beta = beta

    def forward(self, x):
        return torch.nn.functional.softplus(x, beta = self.beta) * (1 - self.gamma) + self.gamma * x
