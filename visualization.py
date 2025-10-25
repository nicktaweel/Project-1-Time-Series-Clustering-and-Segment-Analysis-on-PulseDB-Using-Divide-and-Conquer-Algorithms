import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    def plot_results(self, segments, clusters, closest_pairs, max_subarrays):
        """Visualize results from real VitalDB data"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Cluster distribution
        cluster_sizes = [len(c) for c in clusters]
        axes[0, 0].bar(range(len(cluster_sizes)), cluster_sizes, color='steelblue', alpha=0.7)
        axes[0, 0].set_title('VitalDB Cluster Distribution')
        axes[0, 0].set_xlabel('Cluster ID')
        axes[0, 0].set_ylabel('Number of Segments')

        # 2. Closest pair distances
        distances = [cp['distance'] for cp in closest_pairs]
        axes[0, 1].bar(range(len(distances)), distances, color='coral', alpha=0.7)
        axes[0, 1].set_title('Closest Pair Distances (Real Data)')
        axes[0, 1].set_xlabel('Cluster ID')
        axes[0, 1].set_ylabel('Euclidean Distance')

        # 3. Real VitalDB signal examples
        for i in range(min(3, len(clusters))):
            if clusters[i]:
                # Show actual VitalDB signals
                signal_data = clusters[i][0]['original_signal'][:200]  # First 200 points
                axes[1, 0].plot(signal_data, label=f'Cluster {i}', alpha=0.8)
        axes[1, 0].set_title('Real VitalDB Signal Examples')
        axes[1, 0].set_xlabel('Time (samples)')
        axes[1, 0].set_ylabel('ABP Amplitude')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Maximum subarray analysis
        max_sums = [ms['max_sum'] for ms in max_subarrays[:100]]
        axes[1, 1].hist(max_sums, bins=20, color='lightgreen', alpha=0.7)
        axes[1, 1].set_title('Maximum Subarray Sums (VitalDB)')
        axes[1, 1].set_xlabel('Max Subarray Sum')
        axes[1, 1].set_ylabel('Frequency')

        plt.suptitle('VitalDB Physiological Signal Analysis - Real Data Only', fontsize=16)
        plt.tight_layout()
        plt.savefig('vitaldb_real_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()

    def print_summary(self, segments, clusters, closest_pairs, max_subarrays):
        # Print Summary of VitalDB Analysis
        print("\n" + "-" * 50)
        print("VITALDB ANALYSIS SUMMARY")
        print("-" * 50)
        print(f"Total segments processed: {len(segments)}")
        print(f"Clusters created: {len(clusters)}")
        print(f"Cluster size range: {min(len(c) for c in clusters)}-{max(len(c) for c in clusters)}")

        if closest_pairs:
            avg_distance = np.mean([cp['distance'] for cp in closest_pairs])
            min_distance = min([cp['distance'] for cp in closest_pairs])
            print(f"Average closest pair distance: {avg_distance:.3f}")
            print(f"Minimum closest pair distance: {min_distance:.3f}")

        if max_subarrays:
            max_sum = max([ms['max_sum'] for ms in max_subarrays])
            avg_sum = np.mean([ms['max_sum'] for ms in max_subarrays])
            print(f"Maximum subarray sum: {max_sum:.2f}")
            print(f"Average subarray sum: {avg_sum:.2f}")

        print("-" * 50)