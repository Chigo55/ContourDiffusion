import torch.nn as nn


class SWISH(nn.Sigmoid):
    def __init__(self, *args, **kwargs):
        """
        Initialize the SWISH activation function.

        SWISH is defined as x * sigmoid(x).
        """
        super().__init__(*args, **kwargs)

    def forward(self, input):
        """
        Apply the SWISH activation function to the input tensor.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying SWISH activation.
        """
        return input * super().forward(input=input)