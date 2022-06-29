import torch.nn as nn
import torch.nn.functional as F


class ContentLoss(nn.Module):
    def __init__(
        self,
        target,
    ):
        super(ContentLoss, self).__init__()
        # 기울기를 동적으로 계산하는데 사용되는 tree로부터 타깃 Content를 `분리(detach)` 합니다.
        # 이것은 변수가 아니라 명시된 값입니다.
        # 그렇지 않으면 criterion의 forward 메소드에서 오류가 발생합니다.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input
