def calc_mean_std(feat, eps=1e-5):
    """
    논문의 핵심 아이디어 구현 Instance Normalization을 위한 평균과 표준편차 계산
    Instance Normalization을 구하기 위해 각 이미지의 매 채널에 대하여
    평균(mean)과 표준편차(std) 값을 계산합니다.

    Args:
        feat: Feature Map의 형태
        (N: 배치 크기, C: 채널 크기, H: 높이, W: 너비)
        
        eps: 엡실론(epsilon)은 0으로 나누는 것을 방지하기 위한 작은 상수
        Defaults to 1e-5.

    Returns:
        feat_mean : feature map의 평균
        feat_std : featrure map의 표준편차
    """
    size = feat.size() # Tensor input -> N, C, H, W 높은 차원순
    assert (len(size) == 4)
    N, C = size[:2]
    # 분산
    feat_var = feat.view(N, C, -1).var(dim=2) + eps # vectorize (N, C, H*W)
    # 표준 편차(분산 제곱근 처리)
    feat_std = feat_var.sqrt().view(N, C, 1, 1) # (N, C, H, W)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std