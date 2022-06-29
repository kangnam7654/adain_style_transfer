from pathlib import Path
import sys

ROOT_DIR = Path(__file__).parents[1]
sys.path.append(str(ROOT_DIR))

from utils.calc_mean_std import calc_mean_std


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

if __name__ == '__main__':
    feature_statistics = adaptive_instance_normalization()
