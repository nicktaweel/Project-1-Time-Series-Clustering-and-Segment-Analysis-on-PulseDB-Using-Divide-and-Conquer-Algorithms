from data_loader import DataLoader
from clustering import DivideConquerClustering, ClosestPairFinder
from analysis import KadaneAnalyzer
from visualization import Visualizer
import time
import sys


def main():
    print("VITALDB ANALYSIS - 1000 SEGMENTS × 10 SECONDS")
    print("-" * 60)
    total_start = time.time()

    try:
        # Try loading 1000 segments of 10-second data
        start = time.time()
        loader = DataLoader(n_segments=1000, segment_length=2500)  # 10 seconds at 250Hz
        segments = loader.load_data('VitalDB_Train_Subset.mat')

        segment_duration = segments[0]['duration_seconds'] if segments else 0
        print(f"Segment duration: {segment_duration} seconds")
        print(f"Data loaded: {time.time() - start:.2f}s")

        # Continue with analysis pipeline
        start = time.time()
        clustering = DivideConquerClustering(min_cluster_size=20)
        clusters = clustering.cluster(segments)
        print(f"Clustering: {time.time() - start:.2f}s")

        start = time.time()
        pair_finder = ClosestPairFinder()
        closest_pairs = pair_finder.find_closest_pairs(clusters)
        print(f"Closest pairs: {time.time() - start:.2f}s")

        start = time.time()
        kadane = KadaneAnalyzer()
        max_subarrays = kadane.analyze_segments(segments)
        print(f"Kadane analysis: {time.time() - start:.2f}s")

        start = time.time()
        visualizer = Visualizer()
        visualizer.plot_results(segments, clusters, closest_pairs, max_subarrays)
        visualizer.print_summary(segments, clusters, closest_pairs, max_subarrays)
        print(f"Visualization: {time.time() - start:.2f}s")

        total_time = time.time() - total_start
        print(f"\nTOTAL TIME: {total_time:.2f} seconds")
        print(f"SEGMENTS: {len(segments)} × {segment_duration} seconds")

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()