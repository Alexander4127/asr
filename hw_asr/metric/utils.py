import editdistance

# Don't forget to support cases when target_text == ''


def calc_wer(target_text, predicted_text) -> float:
    target_split = target_text.split(' ')
    pred_split = predicted_text.split(' ')
    if len(target_split) == 0:
        return len(pred_split) > 0
    return editdistance.eval(target_split, pred_split) / len(target_split)


def calc_cer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        return len(predicted_text) > 0
    return editdistance.eval(target_text, predicted_text) / len(target_text)
