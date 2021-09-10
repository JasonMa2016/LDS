"""Generic implementation of multi-layer perceptron."""

from typing import Callable
from typing import Optional
from typing import Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
  """A simple multi-layer perceptron module."""

  def __init__(
      self,
      input_size: int,
      output_sizes: Sequence[int],
      activation_fn: Callable[[], nn.Module] = nn.ReLU,
      dropout_rate: Optional[float] = None,
      activate_final: bool = False,
  ) -> None:
    """Constructs a simple multi-layer-perceptron.

    Args:
      input_size: The size of the input features.
      output_sizes: Sequence of layer sizes.
      w_init: Initializer for Linear weights.
      activation: Activation function to apply between linear layers. Defaults
        to ReLU.
      dropout_rate: Dropout rate to apply, a rate of `None` (the default) or `0`
        means no dropout will be applied.
      activate_final: Whether or not to activate the final layer of the MLP.
    """
    super(MLP, self).__init__()

    layers = list()
    for in_features, out_features in zip(
        [input_size] + list(output_sizes)[:-2],
        output_sizes[:-1],
    ):
      # Fully connected layer.
      layers.append(nn.Linear(in_features, out_features))
      # Activation layer.
      layers.append(activation_fn(inplace=True))
      # (Optional) dropout layer.
      if dropout_rate is not None:
        layers.append(nn.Dropout(p=dropout_rate, inplace=True))
    # Final layer.
    layers.append(nn.Linear(output_sizes[-2], output_sizes[-1]))
    # (Optional) output activation layer.
    if activate_final:
      layers.append(activation_fn(inplace=True))

    self._model = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass from the MLP."""
    return self._model(x)