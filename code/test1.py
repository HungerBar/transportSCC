import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from tqdm import tqdm


class TrafficNetworkRobustnessAnalyzer:
    def __init__(self, traffic_shapefile, scc_shapefile):
        """
        Initialize Traffic Network Robustness Analyzer

        Parameters:
        traffic_shapefile: Path to traffic network shapefile
        scc_shapefile: Path to SCC region shapefile
        """
        self.traffic_gdf = gpd.read_file(traffic_shapefile)
        self.scc_gdf = gpd.read_file(scc_shapefile)
        self.G = None
        self.points_gdf = None
        self.scc_nodes = []

    def extract_city_nodes(self):
        """Extract unique city nodes from traffic data"""
        print("Extracting city nodes...")
        points = []

        for idx, row in self.traffic_gdf.iterrows():
            # Add start point
            points.append(
                {
                    "geometry": Point(row["SourLon"], row["SourLat"]),
                    "sour_lat": row["SourLat"],
                    "sour_lon": row["SourLon"],
                    "name": f"node_{idx}_start",  # Temporary name
                }
            )
            # Add end point
            points.append(
                {
                    "geometry": Point(row["DestLon"], row["DestLat"]),
                    "sour_lat": row["DestLat"],  # Keep field names consistent
                    "sour_lon": row["DestLon"],
                    "name": f"node_{idx}_end",
                }
            )

        # Create GeoDataFrame
        self.points_gdf = gpd.GeoDataFrame(points, crs=self.traffic_gdf.crs)

        # Remove duplicate points (based on coordinates)
        self.points_gdf = self.points_gdf.drop_duplicates(
            subset=["sour_lon", "sour_lat"]
        ).reset_index(drop=True)

        # Assign unique IDs to each node
        self.points_gdf["node_id"] = range(len(self.points_gdf))

        print(f"Extracted {len(self.points_gdf)} unique city nodes")

    def tag_scc_nodes(self, tolerance=1e-3):
        """Tag nodes belonging to SCC regions"""
        print("Tagging SCC nodes...")

        # Ensure coordinate systems match
        self.points_gdf = self.points_gdf.to_crs(self.scc_gdf.crs)

        # Spatial join: determine if points are within SCC polygons
        joined_gdf = gpd.sjoin(
            self.points_gdf, self.scc_gdf, how="left", predicate="within"
        )

        self.points_gdf["is_scc"] = ~joined_gdf.index_right.isna()
        self.points_gdf["scc_id"] = joined_gdf.index_right

        scc_count = self.points_gdf["is_scc"].sum()
        print(f"Tagging complete: {scc_count} nodes belong to SCC regions")

        return scc_count

    def build_traffic_network(self):
        """Build directed traffic network graph"""
        print("Building traffic network graph...")
        self.G = nx.DiGraph()

        # Add nodes
        for idx, row in self.points_gdf.iterrows():
            self.G.add_node(
                row["node_id"],
                geometry=row["geometry"],
                is_scc=row["is_scc"],
                lat=row["sour_lat"],
                lon=row["sour_lon"],
            )

        # Add edges (traffic routes)
        edge_count = 0
        for idx, row in self.traffic_gdf.iterrows():
            src_point = Point(row["SourLon"], row["SourLat"])
            dst_point = Point(row["DestLon"], row["DestLat"])

            # Find start node ID
            src_id = self._find_node_id(src_point)
            # Find end node ID
            dst_id = self._find_node_id(dst_point)

            if src_id is not None and dst_id is not None:
                self.G.add_edge(src_id, dst_id)
                edge_count += 1

        print(
            f"Network construction complete: {len(self.G.nodes)} nodes, {edge_count} edges"
        )

        # Extract SCC node list
        self.scc_nodes = [n for n in self.G.nodes if self.G.nodes[n]["is_scc"]]
        print(f"SCC node count: {len(self.scc_nodes)}")

    def _find_node_id(self, point, tolerance=1e-3):
        """Find node ID by coordinates (with floating point tolerance)"""
        for p_idx, p_row in self.points_gdf.iterrows():
            if point.distance(p_row["geometry"]) < tolerance:
                return p_row["node_id"]
        return None

    def compute_network_metrics(self, G):
        """Calculate network connectivity metrics"""
        if len(G) == 0:
            return 0, 0, 0, []

        # Calculate strongly connected components
        sccs = list(nx.strongly_connected_components(G))
        scc_sizes = [len(scc) for scc in sccs]

        if not scc_sizes:  # Empty graph
            return 0, 0, 0, []

        largest_scc_size = max(scc_sizes)
        num_sccs = len(sccs)

        # Calculate network efficiency metric (average inverse shortest path length)
        try:
            efficiency = self._calculate_efficiency(G)
        except:
            efficiency = 0

        return largest_scc_size, num_sccs, efficiency, scc_sizes

    def _calculate_efficiency(self, G):
        """Calculate network efficiency (connectivity metric)"""
        if len(G) <= 1:
            return 0

        total_efficiency = 0
        node_pairs = 0

        for node in G.nodes():
            try:
                # Calculate shortest paths from this node to all other nodes
                path_lengths = nx.single_source_shortest_path_length(G, node)

                for target, length in path_lengths.items():
                    if node != target and length > 0:
                        total_efficiency += 1.0 / length
                        node_pairs += 1
            except:
                continue

        return total_efficiency / node_pairs if node_pairs > 0 else 0

    def simulate_node_removal_impact(self, damage_threshold=0.5):
        """Simulate impact of removing each SCC node"""
        print("\nStarting SCC node removal impact simulation...")

        # Calculate original network metrics
        (
            original_largest_scc,
            original_num_sccs,
            original_efficiency,
            original_scc_sizes,
        ) = self.compute_network_metrics(self.G)

        print(f"Original network state:")
        print(f"- Total nodes: {len(self.G)}")
        print(
            f"- Largest SCC size: {original_largest_scc} ({original_largest_scc/len(self.G)*100:.1f}%)"
        )
        print(f"- Number of SCCs: {original_num_sccs}")
        print(f"- Network efficiency: {original_efficiency:.4f}")

        results = []

        # Simulate damage for each SCC node
        for i, node in enumerate(tqdm(self.scc_nodes, desc="Simulating node damage")):
            G_removed = self.G.copy()
            G_removed.remove_node(node)

            # Calculate network metrics after removal
            largest_scc_after, num_sccs_after, efficiency_after, scc_sizes_after = (
                self.compute_network_metrics(G_removed)
            )

            # Calculate change rates
            size_ratio = (
                largest_scc_after / original_largest_scc
                if original_largest_scc > 0
                else 0
            )
            efficiency_ratio = (
                efficiency_after / original_efficiency if original_efficiency > 0 else 0
            )

            # Determine if significant damage occurred
            is_significant = size_ratio < damage_threshold

            results.append(
                {
                    "removed_node": node,
                    "largest_scc_size_after": largest_scc_after,
                    "size_ratio": size_ratio,
                    "efficiency_after": efficiency_after,
                    "efficiency_ratio": efficiency_ratio,
                    "num_sccs_after": num_sccs_after,
                    "is_significant": is_significant,
                    "node_lat": self.G.nodes[node]["lat"],
                    "node_lon": self.G.nodes[node]["lon"],
                }
            )

        return results, original_largest_scc, original_efficiency

    def analyze_results(self, results, original_largest_scc, original_efficiency):
        """Analyze simulation results"""
        print("\n" + "=" * 50)
        print("Simulation Results Analysis")
        print("=" * 50)

        if not results:
            print("No results to analyze")
            return

        # Basic statistics
        size_ratios = [r["size_ratio"] for r in results]
        efficiency_ratios = [r["efficiency_ratio"] for r in results]

        significant_damage_count = sum(1 for r in results if r["is_significant"])
        significant_ratio = significant_damage_count / len(results)

        print(f"Damage simulation statistics:")
        print(f"- Number of SCC nodes tested: {len(results)}")
        print(
            f"- Nodes causing significant damage: {significant_damage_count} ({significant_ratio*100:.1f}%)"
        )
        print(f"- Average largest SCC retention rate: {np.mean(size_ratios)*100:.1f}%")
        print(
            f"- Average efficiency retention rate: {np.mean(efficiency_ratios)*100:.1f}%"
        )

        # Find most damaging node
        if results:
            max_damage_node = min(results, key=lambda x: x["size_ratio"])
            print(f"\nMost damaging node:")
            print(f"- Node ID: {max_damage_node['removed_node']}")
            print(
                f"- Location: ({max_damage_node['node_lat']:.4f}, {max_damage_node['node_lon']:.4f})"
            )
            print(
                f"- Largest SCC retention rate: {max_damage_node['size_ratio']*100:.1f}%"
            )
            print(
                f"- Efficiency retention rate: {max_damage_node['efficiency_ratio']*100:.1f}%"
            )

        # Hypothesis verification
        print(f"\nHypothesis verification results:")
        if significant_ratio > 0.5:
            print(
                "✅ Hypothesis confirmed: Over 50% of SCC node damage causes significant network disruption"
            )
            print(
                "   This indicates SCC cities are indeed critical nodes in the network"
            )
        else:
            print(
                "❌ Hypothesis rejected: Most SCC node damage does not cause significant impact"
            )
            print("   May need to redefine SCC or adjust damage threshold")

        return significant_ratio

    def visualize_results(self, results, original_largest_scc):
        """Visualize analysis results"""
        if not results:
            print("Cannot visualize: No result data")
            return

        # Create visualization charts
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Damage level distribution histogram
        size_ratios = [r["size_ratio"] for r in results]
        ax1.hist(size_ratios, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
        ax1.axvline(x=0.5, color="red", linestyle="--", label="Damage threshold (50%)")
        ax1.set_xlabel("Largest SCC retention ratio")
        ax1.set_ylabel("Number of nodes")
        ax1.set_title("SCC Node Damage Level Distribution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Network efficiency change scatter plot
        efficiency_ratios = [r["efficiency_ratio"] for r in results]
        ax2.scatter(size_ratios, efficiency_ratios, alpha=0.6)
        ax2.set_xlabel("Largest SCC retention ratio")
        ax2.set_ylabel("Network efficiency retention ratio")
        ax2.set_title("Network Connectivity vs Efficiency Change")
        ax2.grid(True, alpha=0.3)

        # 3. Damaging node statistics
        damage_levels = [
            "Severe\n(<20%)",
            "Moderate\n(20-50%)",
            "Minor\n(50-80%)",
            "Negligible\n(>80%)",
        ]
        damage_counts = [
            sum(1 for r in results if r["size_ratio"] < 0.2),
            sum(1 for r in results if 0.2 <= r["size_ratio"] < 0.5),
            sum(1 for r in results if 0.5 <= r["size_ratio"] < 0.8),
            sum(1 for r in results if r["size_ratio"] >= 0.8),
        ]
        ax3.bar(
            damage_levels, damage_counts, color=["red", "orange", "yellow", "green"]
        )
        ax3.set_xlabel("Damage level")
        ax3.set_ylabel("Number of nodes")
        ax3.set_title("Node Distribution by Damage Level")

        # 4. Cumulative distribution function
        sorted_ratios = np.sort(size_ratios)
        cdf = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
        ax4.plot(sorted_ratios, cdf, linewidth=2)
        ax4.axvline(x=0.5, color="red", linestyle="--", label="50% damage threshold")
        ax4.set_xlabel("Largest SCC retention ratio")
        ax4.set_ylabel("Cumulative probability")
        ax4.set_title("Damage Level Cumulative Distribution Function")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("scc_damage_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()

    def generate_report(self, results, significant_ratio):
        """Generate detailed analysis report"""
        report = {
            "total_scc_nodes": len(self.scc_nodes),
            "total_nodes": len(self.G),
            "significant_damage_count": sum(1 for r in results if r["is_significant"]),
            "significant_ratio": significant_ratio,
            "avg_size_ratio": np.mean([r["size_ratio"] for r in results]),
            "avg_efficiency_ratio": np.mean([r["efficiency_ratio"] for r in results]),
            "hypothesis_supported": significant_ratio > 0.5,
        }

        # Save detailed results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv("scc_damage_simulation_results.csv", index=False)

        print(f"\nDetailed results saved to 'scc_damage_simulation_results.csv'")

        return report


def main():
    """Main function"""
    traffic_shapefile = (
        "data/transport/china_transport_network.shp"  # Your generated traffic shapefile
    )
    scc_shapefile = "scc/SCC.shp"  # Replace with actual path

    # Create analyzer
    analyzer = TrafficNetworkRobustnessAnalyzer(traffic_shapefile, scc_shapefile)

    try:
        # 1. Data preprocessing
        analyzer.extract_city_nodes()
        scc_count = analyzer.tag_scc_nodes()

        if scc_count == 0:
            print("Warning: No SCC nodes found, please check SCC region data")
            return

        # 2. Build network
        analyzer.build_traffic_network()

        if len(analyzer.G.nodes) == 0:
            print("Error: Failed to build valid traffic network")
            return

        # 3. Simulate damage analysis
        results, original_largest_scc, original_efficiency = (
            analyzer.simulate_node_removal_impact(damage_threshold=0.5)
        )

        # 4. Analyze results
        significant_ratio = analyzer.analyze_results(
            results, original_largest_scc, original_efficiency
        )

        # 5. Visualization
        analyzer.visualize_results(results, original_largest_scc)

        # 6. Generate report
        report = analyzer.generate_report(results, significant_ratio)

        print(f"\nAnalysis complete!")
        print(
            f"Network robustness: {'High' if not report['hypothesis_supported'] else 'Low'}"
        )
        print(
            f"SCC criticality: {'Significant' if report['significant_ratio'] > 0.3 else 'Insignificant'}"
        )

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
