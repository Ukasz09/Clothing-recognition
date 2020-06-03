def calc_correctness(predicted_labels, real_labels):
    correct_qty = 0
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == real_labels[i]:
            correct_qty += 1
    return correct_qty * 100 / len(predicted_labels)
