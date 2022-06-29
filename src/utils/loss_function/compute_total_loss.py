def compute_total_loss(style_score, content_score, style_losses, content_losses, style_weight=1000000, content_weight=1):
    for style_loss in style_losses:
        style_score += style_loss.loss
    for content_loss in content_losses:
        content_score += content_loss.loss

    style_score *= style_weight
    content_score *= content_weight
    total_loss = style_score + content_score
    return total_loss, style_score, content_score
