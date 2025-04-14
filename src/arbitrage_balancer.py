def decide_trade(conf_long, conf_short, threshold=0.6, margin=0.05):
    """
    Compare long and short signal confidence levels and choose the optimal trade direction.

    Args:
        conf_long (float): Confidence score for long trade (0 to 1)
        conf_short (float): Confidence score for short trade (0 to 1)
        threshold (float): Minimum confidence required to take any trade
        margin (float): Minimum difference between confidences to justify one over the other

    Returns:
        int: 1 for LONG, -1 for SHORT, 0 for HOLD
    """

    if conf_short > conf_long + margin and conf_short > threshold:
        return -1  # SHORT
    elif conf_long > conf_short + margin and conf_long > threshold:
        return 1   # LONG
    return 0  # HOLD