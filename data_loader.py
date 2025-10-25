import numpy as np
import h5py
import scipy.signal as signal
import os


class DataLoader:
    def __init__(self, n_segments=1000, segment_length=2500):  # 10 seconds at 250Hz
        self.n_segments = n_segments
        self.segment_length = segment_length
        self.sampling_rate = 250  # Hz

    def load_data(self, file_path):
        # Try to Load 1000 segments of 10-second data
        print(
            f"Loading VitalDB file for {self.n_segments} segments of {self.segment_length / self.sampling_rate} seconds each")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"VitalDB file not found: {file_path}")

        try:
            with h5py.File(file_path, 'r') as mat_file:
                print("File opened successfully")

                # Extract enough signals to create 1000 segments
                segments = self._extract_for_1000_segments(mat_file)

                return segments[:self.n_segments]  # Ensure exactly n_segments

        except Exception as e:
            raise Exception(f"Failed to read file: {e}")

    def _extract_for_1000_segments(self, mat_file):
        # Extract enough data to create 1000 segments
        segments = []

        # Target signals
        target_signals = [
            'Subset/DBP', 'Subset/SBP', 'Subset/ART', 'Subset/ABP',
            'Subset/PPG', 'Subset/ECG', 'DBP', 'SBP', 'ART', 'ABP'
        ]

        print("Extracting signals for 1000 segments")

        all_signals = []
        for signal_path in target_signals:
            try:
                if signal_path in mat_file:
                    dataset = mat_file[signal_path]
                    if isinstance(dataset, h5py.Dataset):
                        print(f"Found {signal_path} - Shape: {dataset.shape}")
                        signals = self._extract_from_dataset(dataset, signal_path)
                        all_signals.extend(signals)

                        # Stop if we have enough source data
                        if self._has_enough_data_for_1000_segments(all_signals):
                            break
            except Exception as e:
                print(f"  - Failed to extract {signal_path}: {e}")

        if not all_signals:
            raise ValueError("No signals found")

        return self._create_1000_segments(all_signals)

    def _extract_from_dataset(self, dataset, source_name):
        # Extract signals from dataset
        signals = []

        try:
            data = dataset[()]

            if hasattr(data, 'shape') and data.shape:
                if data.ndim == 1:
                    # Single long 1D signal
                    if len(data) > self.segment_length:
                        signals.append(data)
                        print(f"  - 1D signal: {len(data)} points (~{len(data) / self.sampling_rate:.1f} seconds)")

                elif data.ndim == 2:
                    # Multiple signals in 2D array
                    if data.shape[0] == 1 and data.shape[1] > self.segment_length:
                        # Single long row
                        signals.append(data[0])
                        print(f"  - Long row: {len(data[0])} points (~{len(data[0]) / self.sampling_rate:.1f} seconds)")
                    elif data.shape[1] == 1 and data.shape[0] > self.segment_length:
                        # Single long column
                        signals.append(data[:, 0])
                        print(
                            f"  - Long column: {len(data[:, 0])} points (~{len(data[:, 0]) / self.sampling_rate:.1f} seconds)")
                    else:
                        # Multiple individual signals
                        num_signals = min(20, data.shape[0])  # Take more signals
                        for i in range(num_signals):
                            if data.shape[1] > self.segment_length:
                                signals.append(data[i])
                        print(f"  - {num_signals} individual signals")

        except Exception as e:
            print(f"  - Extraction error: {e}")

        return signals

    def _has_enough_data_for_1000_segments(self, signals):
        # Check if there is enough source data for 1000 segments
        total_points = sum(len(sig) for sig in signals)
        segments_possible = total_points // self.segment_length
        return segments_possible >= self.n_segments

    def _create_1000_segments(self, vitaldb_signals):
        # Create exactly 1000 segments of 10-second data
        segments = []
        segment_id = 0

        print(f"Creating {self.n_segments} segments from {len(vitaldb_signals)} signals...")

        for signal_idx, signal_data in enumerate(vitaldb_signals):
            if segment_id >= self.n_segments:
                break

            signal_length = len(signal_data)
            segments_possible = signal_length // self.segment_length

            print(f"Signal {signal_idx + 1}: {signal_length} points -> {segments_possible} segments possible")

            if segments_possible == 0:
                continue

            # Use 50% overlap to get more segments from each signal
            step_size = self.segment_length // 2
            max_segments_from_signal = min(100, (signal_length - self.segment_length) // step_size + 1)

            segments_created = 0
            for seg_idx in range(max_segments_from_signal):
                if segment_id >= self.n_segments:
                    break

                start_idx = seg_idx * step_size
                end_idx = start_idx + self.segment_length

                if end_idx > signal_length:
                    break

                raw_segment = signal_data[start_idx:end_idx]

                if self._is_valid_physiological_segment(raw_segment):
                    processed_segment = self._preprocess_signal(raw_segment)

                    segments.append({
                        'id': segment_id,
                        'signal': processed_segment,
                        'original_signal': raw_segment,
                        'source': f"VitalDB_{signal_idx}",
                        'start_index': start_idx,
                        'duration_seconds': self.segment_length / self.sampling_rate,
                        'features': self._extract_features(processed_segment)
                    })
                    segment_id += 1
                    segments_created += 1

            print(f"  - Created {segments_created} segments (total: {segment_id})")

        print(f"Final: {len(segments)} segments created")

        # If we don't have enough, use what we have
        if len(segments) < self.n_segments:
            print(f"Using available {len(segments)} segments (target: {self.n_segments})")

        return segments

    def _is_valid_physiological_segment(self, segment):
        # Validate signal segment
        segment = np.nan_to_num(segment, nan=0.0, posinf=0.0, neginf=0.0)

        if np.all(segment == segment[0]) or np.std(segment) < 0.1:
            return False

        return True

    def _preprocess_signal(self, signal_data):
        # Preprocess 10-second physiological signal
        signal_clean = np.nan_to_num(signal_data, nan=0.0, posinf=0.0, neginf=0.0)

        if len(signal_clean) > 10:
            p05, p95 = np.percentile(signal_clean, [5, 95])
            signal_clean = np.clip(signal_clean, p05, p95)

        signal_detrended = signal.detrend(signal_clean)
        signal_normalized = (signal_detrended - np.mean(signal_detrended)) / (np.std(signal_detrended) + 1e-8)

        return signal_normalized

    def _extract_features(self, signal):
        # Extract features from 10-second segment
        return {
            'mean': float(np.mean(signal)),
            'std': float(np.std(signal)),
            'min': float(np.min(signal)),
            'max': float(np.max(signal)),
            'zero_crossings': int(np.sum(signal[:-1] * signal[1:] < 0)),
            'duration_seconds': len(signal) / self.sampling_rate
        }