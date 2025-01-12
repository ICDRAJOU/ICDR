from torch import nn

from text_net.encoder import CBDE
from text_net.DGRN import DGRN


class AirNet(nn.Module):
    def __init__(self, opt, clip):
        super(AirNet, self).__init__()

        # Restorer
        self.R = DGRN(opt, clip)

        # Encoder
        self.E = CBDE(opt)

    def forward(self, x_query, x_key, text_prompt):
        if self.training:
            fea, logits, labels, inter = self.E(x_query, x_key)

            restored = self.R(x_query, inter, text_prompt)

            return restored, logits, labels
        else:
            fea, inter = self.E(x_query, x_query)

            restored = self.R(x_query, inter, text_prompt)

            return restored
