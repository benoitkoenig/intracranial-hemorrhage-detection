import math

def remove_padding(predictions, output_len):
    padding = (len(predictions) - output_len) / 2
    padding_before = math.ceil(padding)
    padding_after = math.floor(padding)
    return predictions[padding_before : len(predictions) - padding_after]
