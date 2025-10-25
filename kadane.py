def kadane(segment):
    max_sum = float('-inf')
    cur_sum = 0
    start = end = temp = 0

    for i, val in enumerate(segment):
        cur_sum += val
        if cur_sum > max_sum:
            max_sum = cur_sum
            start, end = temp, i
        if cur_sum < 0:
            cur_sum = 0
            temp = i + 1

    return start, end, max_sum

def kadane_analysis(series_list):
    return [kadane(series) for series in series_list]
