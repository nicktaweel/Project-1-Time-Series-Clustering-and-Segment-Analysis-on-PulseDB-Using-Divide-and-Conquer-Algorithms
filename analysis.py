import numpy as np


class KadaneAnalyzer:
    def analyze_segments(self, segments):
        print("Kadane analysis:")
        results = []

        for segment in segments:
            result = self._kadane_fast(segment['signal'])
            results.append({
                'segment_id': segment['id'],
                'max_sum': result['max_sum'],
                'interval': (result['start'], result['end'])
            })

        return results

    def _kadane_fast(self, signal):
        # Optimized Kadane's algorithm
        if len(signal) == 0:
            return {'max_sum': 0, 'start': 0, 'end': 0}

        max_sum = current_sum = signal[0]
        start = end = temp_start = 0

        for i in range(1, len(signal)):
            if current_sum < 0:
                current_sum = signal[i]
                temp_start = i
            else:
                current_sum += signal[i]

            if current_sum > max_sum:
                max_sum = current_sum
                start = temp_start
                end = i

        return {'max_sum': max_sum, 'start': start, 'end': end}