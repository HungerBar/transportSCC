import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec


def load_and_process_data(csv_file):
    """
    Load and process the CSV data
    """
    df = pd.read_csv(csv_file)
    return df


def calculate_damage_levels(df):
    """
    Calculate damage levels based on Largest SCC retention ratio
    """
    # Assuming Damage_Ratio represents the reduction, so retention = 1 - Damage_Ratio
    df["Largest_SCC_Retention"] = 1 - df["Damage_Ratio"]

    # Define damage levels based on retention ratio
    conditions = [
        df["Largest_SCC_Retention"] < 0.2,
        (df["Largest_SCC_Retention"] >= 0.2) & (df["Largest_SCC_Retention"] < 0.5),
        (df["Largest_SCC_Retention"] >= 0.5) & (df["Largest_SCC_Retention"] < 0.8),
        df["Largest_SCC_Retention"] >= 0.8,
    ]

    choices = [
        "Severe (<20%)",
        "Moderate (20-50%)",
        "Minor (50-80%)",
        "Negligible (>80%)",
    ]

    df["Damage_Level"] = np.select(conditions, choices, default="Unknown")
    return df


def calculate_efficiency_change(df):
    """
    Calculate network efficiency change (simulated data for demonstration)
    """
    # This is simulated data - in real analysis, you would calculate actual efficiency metrics
    np.random.seed(42)  # For reproducible results
    df["Efficiency_Retention"] = 0.99 + 0.01 * np.random.randn(len(df))
    df["Efficiency_Retention"] = df["Efficiency_Retention"].clip(0.99, 1.01)
    return df


def create_scc_analysis_plots(df, output_file="scc_analysis_results.png"):
    """
    Create the four-panel SCC analysis plot
    """
    # Set up the figure with 2x2 subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.3)

    # Define colors for damage levels
    colors = {
        "Severe (<20%)": "red",
        "Moderate (20-50%)": "orange",
        "Minor (50-80%)": "yellow",
        "Negligible (>80%)": "green",
    }

    # 1. Top-left: SCC Node Damage Level Distribution (Histogram)
    ax1 = fig.add_subplot(gs[0, 0])
    retention_ratios = df["Largest_SCC_Retention"]

    # Create histogram with density curve
    n, bins, patches = ax1.hist(
        retention_ratios,
        bins=20,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
        density=True,
    )

    # Add density curve
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(retention_ratios)
    x_range = np.linspace(retention_ratios.min(), retention_ratios.max(), 100)
    ax1.plot(x_range, kde(x_range), "r-", linewidth=2, label="Density")

    ax1.set_xlabel("Largest SCC Retention Ratio", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("SCC Node Damage Level Distribution", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Top-right: Network Connectivity vs Efficiency Change (Scatter plot)
    ax2 = fig.add_subplot(gs[0, 1])

    # Group by damage level and plot with different colors
    for level, color in colors.items():
        level_data = df[df["Damage_Level"] == level]
        if len(level_data) > 0:
            ax2.scatter(
                level_data["Largest_SCC_Retention"],
                level_data["Efficiency_Retention"],
                c=color,
                label=level,
                alpha=0.7,
                s=60,
            )

    ax2.set_xlabel("Largest SCC Retention Ratio", fontsize=12)
    ax2.set_ylabel("Network Efficiency Retention Ratio", fontsize=12)
    ax2.set_title(
        "Network Connectivity vs Efficiency Change", fontsize=14, fontweight="bold"
    )
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True, alpha=0.3)

    # 3. Bottom-left: Node Distribution by Damage Level (Bar chart)
    ax3 = fig.add_subplot(gs[1, 0])

    damage_counts = df["Damage_Level"].value_counts()
    # Reorder to match the expected sequence
    ordered_levels = [
        "Severe (<20%)",
        "Moderate (20-50%)",
        "Minor (50-80%)",
        "Negligible (>80%)",
    ]
    damage_counts = damage_counts.reindex(ordered_levels, fill_value=0)

    bars = ax3.bar(
        range(len(damage_counts)),
        damage_counts.values,
        color=[colors[level] for level in damage_counts.index],
        alpha=0.7,
        edgecolor="black",
    )

    ax3.set_xlabel("Damage Level", fontsize=12)
    ax3.set_ylabel("Number of Nodes", fontsize=12)
    ax3.set_title("Node Distribution by Damage Level", fontsize=14, fontweight="bold")
    ax3.set_xticks(range(len(damage_counts)))
    ax3.set_xticklabels(damage_counts.index, rotation=45, ha="right")
    ax3.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    # 4. Bottom-right: Damage Level Cumulative Distribution Function
    ax4 = fig.add_subplot(gs[1, 1])

    # Sort retention ratios and calculate CDF
    sorted_retention = np.sort(retention_ratios)
    cdf = np.arange(1, len(sorted_retention) + 1) / len(sorted_retention)

    ax4.plot(sorted_retention, cdf, "b-", linewidth=2, marker="o", markersize=3)
    ax4.set_xlabel("Largest SCC Retention Ratio", fontsize=12)
    ax4.set_ylabel("Cumulative Probability", fontsize=12)
    ax4.set_title(
        "Damage Level Cumulative Distribution Function", fontsize=14, fontweight="bold"
    )
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.05)

    # Add horizontal lines at key percentiles
    for percentile in [0.25, 0.5, 0.75]:
        ax4.axhline(y=percentile, color="r", linestyle="--", alpha=0.5)
        ax4.text(
            sorted_retention[-1] * 0.9,
            percentile,
            f"{percentile*100}%",
            va="bottom",
            ha="right",
        )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()

    return fig


def generate_summary_statistics(df):
    """
    Generate and print summary statistics
    """
    print("=" * 50)
    print("SCC ANALYSIS SUMMARY STATISTICS")
    print("=" * 50)

    print(f"Total nodes analyzed: {len(df)}")
    print(f"Average SCC Retention Ratio: {df['Largest_SCC_Retention'].mean():.3f}")
    print(f"Median SCC Retention Ratio: {df['Largest_SCC_Retention'].median():.3f}")
    print(f"Standard Deviation: {df['Largest_SCC_Retention'].std():.3f}")

    print("\nDamage Level Distribution:")
    damage_counts = df["Damage_Level"].value_counts()
    for level, count in damage_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {level}: {count} nodes ({percentage:.1f}%)")

    print(f"\nNetwork Efficiency Statistics:")
    print(f"  Average Efficiency Retention: {df['Efficiency_Retention'].mean():.3f}")
    print(
        f"  Efficiency Range: [{df['Efficiency_Retention'].min():.3f}, {df['Efficiency_Retention'].max():.3f}]"
    )


def main():
    """
    Main function to run the SCC analysis
    """
    # Replace with your CSV file path
    csv_file = "attack_simulation_results.csv"  # Update this path

    try:
        # Load and process data
        print("Loading data...")
        df = load_and_process_data(csv_file)

        # Calculate damage levels and efficiency changes
        df = calculate_damage_levels(df)
        df = calculate_efficiency_change(df)

        # Display basic info about the data
        print(f"Data loaded successfully: {len(df)} records")
        print("\nFirst few rows of processed data:")
        print(
            df[
                [
                    "Attacked_City",
                    "Largest_SCC_Retention",
                    "Damage_Level",
                    "Efficiency_Retention",
                ]
            ].head()
        )

        # Generate summary statistics
        generate_summary_statistics(df)

        # Create and save the four-panel plot
        print("\nGenerating visualization...")
        fig = create_scc_analysis_plots(df, "scc_analysis_four_panel.png")

        # Save processed data with damage levels
        output_csv = "scc_analysis_with_damage_levels.csv"
        df.to_csv(output_csv, index=False)
        print(f"\nProcessed data saved to: {output_csv}")

    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        print("Please update the csv_file path in the main() function.")
    except Exception as e:
        print(f"Error processing data: {e}")


if __name__ == "__main__":
    main()
