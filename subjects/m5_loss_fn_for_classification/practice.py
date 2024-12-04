import torch
import torch.nn as nn
import torch.optim as optim


def line_break():
    print('-' * 16)


class BCELoss():
    def __init__(self, y_prob, y_true) -> None:
        self.y_prob, self.y_true = y_prob, y_true
        self.criterion = nn.BCELoss()

    def compute_loss(self):
        loss = self.criterion(self.y_prob, self.y_true)

        return loss


class BCELogitLoss():
    def __init__(self, y_logits, y_true) -> None:
        self.y_logits, self.y_true = y_logits, y_true
        self.criterion = nn.BCEWithLogitsLoss()

    def compute_loss(self):
        loss = self.criterion(self.y_logits, self.y_true)

        return loss


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, input, target):
        loss = torch.clamp(1 - input * target, min=0)

        return loss.mean()


class HingeLogitLoss():
    def __init__(self, y_pred, y_true) -> None:
        self.y_pred, self.y_true = y_pred, y_true
        self.criterion = HingeLoss()

    def compute_loss(self):
        loss = self.criterion(self.y_pred, self.y_true)

        return loss


class SparseCategoricalCrossEntropyLoss():
    def __init__(self, y_logits, y_true) -> None:
        self.y_logits, self.y_true = y_logits, y_true
        self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self):
        loss = self.criterion(self.y_logits, self.y_true)

        return loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes: int, smoothing: float = 0.1) -> None:
        super(LabelSmoothingLoss, self).__init__()
        self.classes, self.smoothing = classes, smoothing

    def forward(self, pred, target):
        # Convert target labels to one-hot encoding
        target_one_hot = nn.functional.\
            one_hot(target, num_classes=self.classes).float()
        print(target_one_hot)
        # Apply Label Smoothing
        target_smooth = self._compute_smoothing_target(target_one_hot)
        print(target_smooth)
        # Calculate CE loss with the smoothed probabilities
        log_prob = nn.functional.log_softmax(pred, dim=-1)
        print(log_prob)
        loss = self._compute_CE(log_prob=log_prob, target_smooth=target_smooth)

        return loss.mean()

    def _compute_smoothing_target(self, target_one_hot):
        smooth = target_one_hot * (1 - self.smoothing) + \
                self.smoothing / self.classes

        return smooth

    def _compute_CE(self, log_prob, target_smooth):
        return -1 * torch.sum(log_prob * target_smooth, dim=-1)


class BCELossInMultiLabelClassification():
    def __init__(self, y_pred_logits, y_true) -> None:
        self.y_pred_logits, self.y_true = y_pred_logits, y_true
        self.criterion = nn.BCEWithLogitsLoss()

    def compute_loss(self):
        loss = self.criterion(self.y_pred_logits, self.y_true)

        return loss


class PairwiseRankingLoss(nn.Module):
    def __init__(self) -> None:
        super(PairwiseRankingLoss, self).__init__()

    def forward(self, y_pred, y_true):
        loss = 0
        num_pairs = 0

        for i in range(y_pred.size(0)):
            # positive labels
            pos_indices = torch.where(y_true[i] == 1)[0]
            # negative labels
            neg_indices = torch.where(y_true[i] == 0)[0]

            for pos in pos_indices:
                for neg in neg_indices:
                    loss += torch.clamp(1 - y_pred[i, pos] + y_pred[i, neg],
                                        min=0)
                    num_pairs += 1

        if num_pairs > 0:
            loss /= num_pairs

        return loss


class SimpleLinearModel(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super(SimpleLinearModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class BCEAndPairWiseCombineLoss(nn.Module):
    def __init__(self, alpha) -> None:
        super().__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.pairwise_loss = PairwiseRankingLoss()
        self.all_losses: list[torch.Tensor] = []

    def forward(self, pred, target):
        bce_loss = self.bce_loss(pred, target)
        pr_loss = self.pairwise_loss(pred, target)
        total_loss = self.alpha * bce_loss + (1 - alpha) * pr_loss
        self.all_losses.append(bce_loss)
        self.all_losses.append(pr_loss)

        return total_loss

if __name__ == "__main__":
    # Predicted probabilities
    y_pred_prob = torch.tensor([0.8, 0.1, 0.6], requires_grad=True)
    y_true = torch.tensor([1.0, 0.0, 1.0])  # Actual labels
    bceLoss = BCELoss(y_prob=y_pred_prob, y_true=y_true)
    loss = bceLoss.compute_loss()

    print(f'Binary Cross-Entropy Loss: {loss.item()}')

    # Backward pass
    loss.backward()
    print(f'Gradients after backward: {y_pred_prob.grad}')

    line_break()
    # Predicted logits
    y_pred_logits = torch.tensor([1.5, -1.0, 0.2], requires_grad=True)
    y_true = torch.tensor([1.0, 0.0, 1.0])  # Actual labels

    bceLogitLoss = BCELogitLoss(y_logits=y_pred_logits, y_true=y_true)
    loss = bceLogitLoss.compute_loss()

    print(f'Binary Cross-Entropy with Logits Loss: {loss.item()}')

    # Backward pass
    loss.backward()
    print(f'Gradients after backward: {y_pred_logits.grad}')

    y_pred = torch.tensor([0.9, -0.1, 0.8, -0.4], requires_grad=True)
    y_true = torch.tensor([1, -1, 1, -1], dtype=torch.float32) 

    line_break()
    y_pred = torch.tensor([0.9, -0.1, 0.8, -0.4], requires_grad=True)
    y_true = torch.tensor([1, -1, 1, -1], dtype=torch.float32)

    hingeLoss = HingeLogitLoss(y_pred=y_pred, y_true=y_true)

    # Compute loss
    loss = hingeLoss.criterion(y_pred, y_true)
    print("Hinge Loss:", loss.item())

    # Optimize using Gradient Descent
    optimizer = optim.SGD([y_pred], lr=0.1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Gradients after backward: {y_pred.grad}')

    line_break()
    # Cross Entropy Loss
    # Assume probability distributions
    # f(x): Predicted distribution from the model
    f_x = torch.tensor([0.4, 0.1, 0.3, 0.2])

    # p(x): True distribution (one-hot vector)
    p_x = torch.tensor([0.0, 1.0, 0.0, 0.0])

    # Calculate Cross-Entropy Loss manually
    cross_entropy_loss = -torch.sum(p_x * torch.log(f_x + 1e-10))
    print("Manual Cross-Entropy Loss:", cross_entropy_loss.item())

    line_break()
    # Sparse Categorical Cross Entropy (nn.CrossEntropyLoss in PyTorch)
    # Example logits for a 3-class classification (batch size = 2)
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])

    # True class labels, integer labels
    labels = torch.tensor([0, 2])

    SCCELoss = SparseCategoricalCrossEntropyLoss(y_logits=logits,
                                                 y_true=labels)
    loss = SCCELoss.compute_loss()

    print(f"Cross-Entropy Loss: {loss.item()}")
    # loss.backward() --> this will raise error due to requires_grad variable is not used is logits tensor
    # print(logits.grad)

    line_break()
    num_classes = 5
    label_smoothing = 0.1
    batch_size = 3

    criterion = LabelSmoothingLoss(classes=num_classes,
                                   smoothing=label_smoothing)
    # Unnormalized values (logits)
    pred = torch.randn(batch_size, num_classes)
    print(pred)
    # Ground truth labels
    target = torch.tensor([1, 0, 3])

    loss = criterion(pred, target)
    print(f'Label smoothing Loss: {loss.item()}')

    line_break()
    # BCE Loss in Multi-Label Classification
    # Number of samples and labels
    num_samples = 4
    num_labels = 3

    # Model predictions in logits form (not passed through sigmoid)
    y_pred_logits = torch.tensor([
                                    [0.8, -1.2, 1.5],
                                    [1.2, 0.3, -0.8],
                                    [0.2, 2.0, -1.5],
                                    [0.7, -0.5, 1.0]
                                ])

    # Ground truth labels, each row represents labels for a sample
    # (1 indicates the label is present, 0 indicates the label is absent)
    y_true = torch.tensor([
                        [1, 0, 1],
                        [0, 1, 0],
                        [1, 1, 0],
                        [0, 0, 1]]).float()

    bceMultiLabelLoss = BCELossInMultiLabelClassification(
        y_pred_logits=y_pred_logits,
        y_true=y_true
    )

    loss = bceMultiLabelLoss.compute_loss()
    print(f"Binary CE Loss: {loss.item()}")

    line_break()
    torch.manual_seed(0)
    num_samples = 10
    num_labels = 5
    X = torch.randn(num_samples, 4)
    Y = (torch.rand(num_samples, num_labels) > 0.5).float()

    model = SimpleLinearModel(input_dim=4, output_dim=num_labels)
    y_pred = model(X)
    loss = PairwiseRankingLoss()(y_pred, Y)
    print(f'Pairwise Ranking Loss: {loss.item()}')

    line_break()
    # BCE Loss + PR Loss
    num_samples = 4
    num_labels = 3
    alpha = 0.5

    # Predicted logits and true labels
    y_pred = torch.randn(num_samples, num_labels)  # Random logits
    y_true = torch.randint(0, 2, (num_samples, num_labels)).float()  # Random binary labels

    # Print test data
    print("Predicted logits (y_pred):\n", y_pred)
    print("True labels (y_true):\n", y_true)

    combinedLoss = BCEAndPairWiseCombineLoss(alpha=alpha)
    combined_loss = combinedLoss(y_pred, y_true)
    bce_loss, pr_loss = combinedLoss.all_losses

    print(f"BCE Loss: {bce_loss.item()}")
    print(f"Pairwise Ranking Loss: {pr_loss.item()}")
    print(f"Combined Loss (alpha=0.5): {combined_loss.item()}")
