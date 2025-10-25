import numpy as np
from scipy.spatial.distance import euclidean


class DivideConquerClustering:
    def __init__(self, min_cluster_size=5, max_depth=5):
        self.min_cluster_size = min_cluster_size
        self.max_depth = max_depth

    def cluster(self, segments):
        # Clustering that handles small numbers of segments
        print(f"Clustering {len(segments)} segments")

        if len(segments) <= self.min_cluster_size:
            print("Too few segments for clustering, returning single cluster")
            return [segments]

        clusters = self._split_recursive(segments)
        clusters = [c for c in clusters if len(c) > 0]

        print(f"Created {len(clusters)} clusters")
        for i, cluster in enumerate(clusters):
            print(f"  Cluster {i}: {len(cluster)} segments")

        return clusters

    def _split_recursive(self, segments, depth=0):
        if len(segments) <= self.min_cluster_size or depth >= self.max_depth:
            return [segments]

        # Use features for splitting
        features = []
        for s in segments:
            sig = s['signal']
            features.append([
                np.mean(sig),
                np.std(sig),
                np.sum(sig[:-1] * sig[1:] < 0)  # zero crossings
            ])

        features = np.array(features)
        split_dim = np.argmax(np.var(features, axis=0))
        split_value = np.median(features[:, split_dim])

        left = [s for s, f in zip(segments, features) if f[split_dim] <= split_value]
        right = [s for s, f in zip(segments, features) if f[split_dim] > split_value]

        clusters = []
        if left:
            clusters.extend(self._split_recursive(left, depth + 1))
        if right:
            clusters.extend(self._split_recursive(right, depth + 1))

        return clusters


class ClosestPairFinder:
    def find_closest_pairs(self, clusters):
        # Find closest pairs with proper error handling
        print("Computing closest pairs")
        closest_pairs = []

        for i, cluster in enumerate(clusters):
            if len(cluster) < 2:
                print(f"  Cluster {i}: Too few segments for closest pair")
                continue

            min_dist = float('inf')
            closest_pair = None

            # Compare all pairs in the cluster
            for j in range(len(cluster)):
                for k in range(j + 1, len(cluster)):
                    try:
                        # Ensure we have proper arrays
                        sig1 = np.array(cluster[j]['signal'], dtype=float)
                        sig2 = np.array(cluster[k]['signal'], dtype=float)

                        # Use downsampled signals for speed
                        if len(sig1) > 100:
                            sig1 = sig1[::10]
                        if len(sig2) > 100:
                            sig2 = sig2[::10]

                        dist = euclidean(sig1, sig2)

                        if dist < min_dist:
                            min_dist = dist
                            closest_pair = (cluster[j], cluster[k])
                    except Exception as e:
                        print(f"    Distance computation failed: {e}")
                        continue

            if closest_pair:
                closest_pairs.append({
                    'cluster_id': i,
                    'pair': closest_pair,
                    'distance': min_dist
                })
                print(f"  Cluster {i}: closest pair distance = {min_dist:.3f}")

        return closest_pairs