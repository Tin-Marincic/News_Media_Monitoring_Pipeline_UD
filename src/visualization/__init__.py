"""
Visualization package for the News Media Monitoring Pipeline.
Re-exports static and interactive chart functions for easier imports.
"""

from src.visualization.static_charts import (
    plot_top_categories_bar,
    plot_document_type_counts,
    plot_rating_distribution,
    plot_rating_by_category_boxplot,
    plot_popularity_vs_content_length_scatter,
    plot_average_rating_over_years,
    plot_numeric_correlation_heatmap,
    plot_news_dashboard_subplots,
    generate_all_static_charts,
)

from src.visualization.interactive_charts import (
    interactive_popularity_vs_content_length,
    interactive_top_categories_bar,
    interactive_records_over_years,
    interactive_rating_by_category_boxplot,
    interactive_news_multi_layout,
    generate_all_interactive_charts,
)

from src.visualization.chart_generator import (
    generate_visualizations,
    load_visualization_data,
)

__all__ = [
    "plot_top_categories_bar",
    "plot_document_type_counts",
    "plot_rating_distribution",
    "plot_rating_by_category_boxplot",
    "plot_popularity_vs_content_length_scatter",
    "plot_average_rating_over_years",
    "plot_numeric_correlation_heatmap",
    "plot_news_dashboard_subplots",
    "generate_all_static_charts",
    "interactive_popularity_vs_content_length",
    "interactive_top_categories_bar",
    "interactive_records_over_years",
    "interactive_rating_by_category_boxplot",
    "interactive_news_multi_layout",
    "generate_all_interactive_charts",
    "generate_visualizations",
    "load_visualization_data",
]