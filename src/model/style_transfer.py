def calc_mean_std(feat, eps=1e-6):
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


def adaptive_instance_normalization(content_feat=None, style_feat=None):
    """
    논문에서 제시한 AdaIN을 구현
    AdaIN은 content feature의 스타일을 style feature의 스타일로 변경하는 연산
    Args:
        content_feat (_type_): _description_
        style_feat (_type_): _description_

    Returns:
        _type_: _description_
    """
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    # 평균(mean)과 표준편차(std)를 이용하여 정규화 수행
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    # 정규화 이후에 style feature의 statistics를 가지도록 설정
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def AdaIN_transfer(encoder, decoder, content_input, style_input, alpha=1.0):
    # assert (0.0 <= alpha <= 1.0)
    content_feature = encoder(content_input)
    style_feature = encoder(style_input)
    feature = adaptive_instance_normalization(content_feature, style_feature)
    feature = feature * alpha + content_feature * (1 - alpha) # Alpha가 1에 가까울수록 스타일이 진해짐
    return decoder(feature)