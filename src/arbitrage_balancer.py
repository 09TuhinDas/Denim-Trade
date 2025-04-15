
def decide_trade(conf_long, conf_short, threshold=0.6, margin=0.05):
    """
    Decide trade direction based on calibrated confidence and margin gap.
    Returns 1 (LONG), -1 (SHORT), or 0 (HOLD).
    """
    if conf_long >= threshold and (conf_long - conf_short) > margin:
        return 1
    elif conf_short >= threshold and (conf_short - conf_long) > margin:
        return -1
    return 0
