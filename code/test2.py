import csv
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon


def load_and_validate_data(scc_shp_path, routes_shp_path):
    """
    Load and validate SCC (polygon) and transportation routes (line) Shapefile data.
    """
    print("Loading data...")
    try:
        # Load data
        scc_gdf = gpd.read_file(scc_shp_path)
        routes_gdf = gpd.read_file(routes_shp_path)

        print("=" * 50)
        print("SCC Shapefile field information:")
        print(scc_gdf.dtypes)
        print(f"\nSCC data sample (first 3 rows):")
        print(scc_gdf[["NAME", "id", "in_count", "out_count", "total_degr"]].head(3))

        print("\n" + "=" * 50)
        print("Transportation routes Shapefile field information:")
        print(routes_gdf.dtypes)

        return scc_gdf, routes_gdf

    except Exception as e:
        print(f"Error: Failed to load Shapefile - {e}")
        return None, None


def build_transportation_network(routes_gdf, scc_gdf):
    """
    Build transportation network graph using spatial queries.
    Uses start and end coordinates of roads to determine which cities they belong to via spatial queries.
    """
    G = nx.DiGraph()
    problematic_edges = []

    print("\n" + "=" * 50)
    print("Building transportation network through spatial queries...")

    # First, add all city names from SCC file as nodes to the graph
    valid_cities_from_scc = set(scc_gdf["NAME"].dropna().unique())
    for city in valid_cities_from_scc:
        G.add_node(city)
    print(f"Added {len(valid_cities_from_scc)} city nodes from SCC file.")

    # Statistical variables for successful edge matching
    successful_edges = 0

    for idx, row in routes_gdf.iterrows():
        try:
            # Get start and end coordinates of the road
            start_lon, start_lat = row["SourLon"], row["SourLat"]
            end_lon, end_lat = row["DestLon"], row["DestLat"]

            # Create Point geometry objects for start and end points
            start_point = Point(start_lon, start_lat)
            end_point = Point(end_lon, end_lat)

            # Find the city where the start point is located
            start_city = None
            for _, city_row in scc_gdf.iterrows():
                city_geom = city_row["geometry"]
                if city_geom.contains(start_point):
                    start_city = city_row["NAME"]
                    break

            # Find the city where the end point is located
            end_city = None
            for _, city_row in scc_gdf.iterrows():
                city_geom = city_row["geometry"]
                if city_geom.contains(end_point):
                    end_city = city_row["NAME"]
                    break

            # If both start and end points are in known cities and not the same city, add edge
            if start_city and end_city and start_city != end_city:
                # Add edge (from start city to end city)
                G.add_edge(start_city, end_city)
                successful_edges += 1

                # Print progress every 100 successfully matched edges
                if successful_edges % 100 == 0:
                    print(f"Successfully added {successful_edges} edges...")

            elif not start_city and not end_city:
                problematic_edges.append(
                    f"Index {idx}: Start and end points not in any city"
                )
            elif not start_city:
                problematic_edges.append(f"Index {idx}: Start point not in any city")
            elif not end_city:
                problematic_edges.append(f"Index {idx}: End point not in any city")
            elif start_city == end_city:
                # Paths within the same city, can be ignored or specially processed
                pass

        except Exception as e:
            problematic_edges.append(f"Index {idx}: Processing error - {e}")

    print(
        f"Network construction completed. Successfully added {successful_edges} edges."
    )
    if problematic_edges:
        print(f"There are {len(problematic_edges)} edges with processing issues.")
        if len(problematic_edges) <= 10:  # Only show first 10 problems
            for problem in problematic_edges[:10]:
                print(f"  - {problem}")

    return G, problematic_edges


def analyze_network_impact(G, scc_gdf, threshold=42):
    """
    Simulate attacks on SCC cities and analyze impact on the network.
    """
    print("\n" + "=" * 50)
    print("Starting attack simulation and analysis...")

    # Get nodes with degree greater than threshold from SCC shapefile
    high_degree_cities = scc_gdf[scc_gdf["total_degr"] > threshold]["NAME"].tolist()
    # Ensure these cities exist in our constructed network G
    existing_high_degree_cities = [
        city for city in high_degree_cities if city in G.nodes()
    ]
    print(
        f"Number of cities with degree > {threshold} in SCC file: {len(high_degree_cities)}"
    )
    print(
        f"Number of key cities actually existing in the network graph: {len(existing_high_degree_cities)}"
    )

    if len(existing_high_degree_cities) == 0:
        print("Warning: No key cities found for attack simulation.")
        return pd.DataFrame()

    attack_results = []
    original_city_count = len(existing_high_degree_cities)

    for index, target_city in enumerate(existing_high_degree_cities):
        G_attacked = G.copy()
        G_attacked.remove_node(target_city)

        # Calculate number of cities still meeting high degree threshold after attack
        remaining_high_degree_count = 0
        for node in G_attacked.nodes():
            total_degree = G_attacked.in_degree(node) + G_attacked.out_degree(node)
            if total_degree > threshold:
                remaining_high_degree_count += 1

        damage_ratio = (
            original_city_count - remaining_high_degree_count
        ) / original_city_count
        attack_results.append(
            {
                "Attacked_City": target_city,
                "Remaining_SCC_Count": remaining_high_degree_count,
                "SCC_Reduction": original_city_count - remaining_high_degree_count,
                "Damage_Ratio": damage_ratio,
            }
        )

    results_df = pd.DataFrame(attack_results)
    return results_df


def save_results_to_csv(
    results_df, network_info, filename="attack_simulation_results.csv"
):
    """
    Save experiment results to CSV file.
    """
    try:
        # Save attack results
        results_df.to_csv(filename, index=False)
        print(f"Attack results saved to {filename}")

        # Save network information to separate CSV
        network_df = pd.DataFrame([network_info])
        network_df.to_csv("network_information.csv", index=False)
        print("Network information saved to network_information.csv")

        return True
    except Exception as e:
        print(f"Error saving results to CSV: {e}")
        return False


def visualize_results(results_df, G, output_dir="results"):
    """
    Create visualizations of the experiment results.
    """
    try:
        # Create results directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. Bar chart of damage ratios
        plt.figure(figsize=(12, 6))
        top_cities = results_df.nlargest(15, "Damage_Ratio")
        bars = plt.bar(range(len(top_cities)), top_cities["Damage_Ratio"])
        plt.xlabel("Cities")
        plt.ylabel("Damage Ratio")
        plt.title("Top 15 Cities by Network Damage Ratio")
        plt.xticks(
            range(len(top_cities)), top_cities["Attacked_City"], rotation=45, ha="right"
        )

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/damage_ratio_barchart.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 2. Histogram of damage ratios
        plt.figure(figsize=(10, 6))
        plt.hist(results_df["Damage_Ratio"], bins=20, edgecolor="black", alpha=0.7)
        plt.xlabel("Damage Ratio")
        plt.ylabel("Frequency")
        plt.title("Distribution of Network Damage Ratios")
        plt.grid(True, alpha=0.3)
        plt.savefig(
            f"{output_dir}/damage_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 3. Scatter plot: SCC Reduction vs Damage Ratio
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df["SCC_Reduction"], results_df["Damage_Ratio"], alpha=0.6)
        plt.xlabel("SCC Reduction")
        plt.ylabel("Damage Ratio")
        plt.title("SCC Reduction vs Damage Ratio")
        plt.grid(True, alpha=0.3)
        plt.savefig(
            f"{output_dir}/scc_vs_damage_scatter.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 4. Simple network visualization (if not too large)
        if G.number_of_nodes() <= 100:  # Only visualize if network is not too large
            plt.figure(figsize=(15, 10))
            pos = nx.spring_layout(G, k=1, iterations=50)
            nx.draw_networkx_nodes(G, pos, node_size=50, node_color="lightblue")
            nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.5, arrowsize=10)
            nx.draw_networkx_labels(G, pos, font_size=8)
            plt.title("Transportation Network Structure")
            plt.axis("off")
            plt.savefig(
                f"{output_dir}/network_structure.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

        print(f"Visualizations saved to {output_dir} directory")
        return True

    except Exception as e:
        print(f"Error creating visualizations: {e}")
        return False


def main():
    # Please replace with your actual file paths
    scc_shapefile_path = "scc/SCC.shp"
    routes_shapefile_path = "data/transport/china_transport_network.shp"

    # 1. Load and validate data
    scc_gdf, routes_gdf = load_and_validate_data(
        scc_shapefile_path, routes_shapefile_path
    )
    if scc_gdf is None or routes_gdf is None:
        print("Data loading failed. Please check file paths.")
        return

    # 2. Build transportation network
    G, problematic_edges = build_transportation_network(routes_gdf, scc_gdf)

    if G.number_of_edges() == 0:
        print("Network has 0 edges. Analysis cannot be performed.")
        return

    print(
        f"\nOriginal transportation network construction completed. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}"
    )

    # 3. Analyze network robustness
    results_df = analyze_network_impact(G, scc_gdf, threshold=42)

    if not results_df.empty:
        # 4. Save results to CSV
        network_info = {
            "Total_Nodes": G.number_of_nodes(),
            "Total_Edges": G.number_of_edges(),
            "Problematic_Edges_Count": len(problematic_edges),
            "Cities_Simulated": len(results_df),
        }

        save_results_to_csv(results_df, network_info, "attack_simulation_results.csv")

        # 5. Create visualizations
        visualize_results(results_df, G, "results")

        # 6. Output analysis summary
        print("\n" + "=" * 50)
        print("Attack Simulation Results Summary:")
        print(f"Cities simulated for attack: {len(results_df)}")
        print(f"Statistics of remaining SCC nodes after attack:")
        print(results_df["Remaining_SCC_Count"].describe())

        max_impact = results_df.loc[results_df["Damage_Ratio"].idxmax()]
        print(f"\nNode causing maximum damage: '{max_impact['Attacked_City']}'")
        print(f"  Damage ratio: {max_impact['Damage_Ratio']:.2%}")

        # Determine if it constitutes "massive damage"
        damage_threshold = 0.30
        high_impact_nodes = results_df[results_df["Damage_Ratio"] >= damage_threshold]
        print(
            f"\nTotal of {len(high_impact_nodes)} nodes with damage ratio >= {damage_threshold:.0%}."
        )
        if len(high_impact_nodes) > 0:
            print(
                "Damage from these key nodes significantly affects network connectivity:"
            )
            for idx, row in high_impact_nodes.iterrows():
                print(f"  - {row['Attacked_City']}: {row['Damage_Ratio']:.2%}")

        # Save high impact nodes to separate CSV
        high_impact_nodes.to_csv("high_impact_nodes.csv", index=False)
        print("High impact nodes saved to high_impact_nodes.csv")

    else:
        print("Attack simulation did not produce valid results.")


if __name__ == "__main__":
    main()
