import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Learn a correction to the incoming board features."""

    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(x + self.layers(x))


class GlobalContext(nn.Module):
    """Use a whole-board summary to gate each feature channel."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(1, channels // 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class GomokuNetwork(nn.Module):
    def __init__(
        self,
        channels=32,
        value_hidden=64,
        policy_kernel=1,
        architecture="cnn",
        residual_blocks=3,
    ):
        super().__init__()

        if policy_kernel not in (1, 3):
            raise ValueError("Policy kernel must be 1 or 3")
        if architecture not in ("cnn", "resnet_global"):
            raise ValueError("Unknown network architecture")
        if (
            isinstance(residual_blocks, bool)
            or not isinstance(residual_blocks, int)
            or residual_blocks < 1
        ):
            raise ValueError("Residual blocks must be a positive integer")

        # Preserve the original module names and shapes for old checkpoints.
        if architecture == "cnn":
            self.trunk = nn.Sequential(
                nn.Conv2d(3, channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.trunk = nn.Sequential(
                nn.Conv2d(3, channels, kernel_size=3, padding=1),
                nn.ReLU(),
                *(ResidualBlock(channels) for _ in range(residual_blocks)),
                GlobalContext(channels),
            )

        self.policy_head = nn.Sequential(
            nn.Conv2d(
                channels,
                1,
                kernel_size=policy_kernel,
                padding=policy_kernel // 2,
            ),
            nn.Flatten(start_dim=1),
        )

        if architecture == "cnn":
            self.value_head = nn.Sequential(
                nn.Conv2d(channels, 1, kernel_size=1),
                nn.Flatten(start_dim=1),
                nn.Linear(19 * 19, value_hidden),
                nn.ReLU(),
                nn.Linear(value_hidden, 1),
                nn.Tanh(),
            )
        else:
            self.value_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(start_dim=1),
                nn.Linear(channels, value_hidden),
                nn.ReLU(),
                nn.Linear(value_hidden, 1),
                nn.Tanh(),
            )

    def forward(self, x):
        features = self.trunk(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)

        return policy_logits, value


def masked_policy(policy_logits, legal_mask):
    if policy_logits.ndim != 2:
        raise ValueError("Policy logits must have shape (batch, actions)")

    if legal_mask.shape != policy_logits.shape:
        raise ValueError("Legal mask must have the same shape as logits")

    legal_mask = legal_mask.to(
        device=policy_logits.device,
        dtype=torch.bool,
    )

    if not legal_mask.any(dim=1).all().item():
        raise ValueError("Each position must have at least one legal action")

    masked_logits = policy_logits.masked_fill(
        ~legal_mask,
        float("-inf"),
    )

    return torch.softmax(masked_logits, dim=1)


if __name__ == "__main__":
    model = GomokuNetwork(
        policy_kernel=3,
        architecture="resnet_global",
        residual_blocks=3,
    )
    x = torch.randn(1, 3, 19, 19)

    with torch.no_grad():
        policy, value = model(x)

    print("Policy shape:", policy.shape)
    print("Value shape:", value.shape)
