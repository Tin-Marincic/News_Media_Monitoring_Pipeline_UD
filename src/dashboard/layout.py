import dash_bootstrap_components as dbc
from dash import dcc, html

from src.dashboard.data_access import (
    load_dashboard_data,
    get_available_categories,
    get_available_document_types,
    get_year_range,
    get_summary_metrics,
)


DASHBOARD_DF = load_dashboard_data()
CATEGORIES = get_available_categories(DASHBOARD_DF)
DOCUMENT_TYPES = get_available_document_types(DASHBOARD_DF)
YEAR_MIN, YEAR_MAX = get_year_range(DASHBOARD_DF)
YEAR_SLIDER_MAX = YEAR_MAX if YEAR_MAX > YEAR_MIN else YEAR_MIN + 1
METRICS = get_summary_metrics(DASHBOARD_DF)


CARD_STYLE = {
    "background": "linear-gradient(135deg, #111827, #1f2937)",
    "border": "1px solid #374151",
    "borderRadius": "16px",
    "boxShadow": "0 12px 28px rgba(0,0,0,0.28)",
}

GRAPH_CARD_STYLE = {
    "background": "#111827",
    "border": "1px solid #374151",
    "borderRadius": "18px",
    "padding": "16px",
    "boxShadow": "0 12px 28px rgba(0,0,0,0.30)",
}

PAGE_STYLE = {
    "background": "linear-gradient(180deg, #020617 0%, #0f172a 48%, #111827 100%)",
    "minHeight": "100vh",
    "color": "#f8fafc",
    "paddingBottom": "40px",
}


def metric_card(title: str, value, subtitle: str, card_id: str):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="text-uppercase small text-secondary fw-semibold"),
                html.H3(
                    value,
                    id=card_id,
                    className="fw-bold mt-2 mb-1",
                    style={"color": "#f8fafc"},
                ),
                html.Div(subtitle, className="small", style={"color": "#94a3b8"}),
            ]
        ),
        style=CARD_STYLE,
        className="h-100",
    )


def graph_card(title: str, graph_id: str, description: str | None = None, height: str = "420px"):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.H5(title, className="fw-bold mb-1", style={"color": "#f8fafc"}),
                        html.P(
                            description or "",
                            className="mb-3",
                            style={"color": "#94a3b8", "fontSize": "0.92rem"},
                        ),
                    ]
                ),
                dcc.Loading(
                    dcc.Graph(
                        id=graph_id,
                        config={"displayModeBar": True, "responsive": True},
                        style={"height": height},
                    ),
                    type="circle",
                ),
            ]
        ),
        style=GRAPH_CARD_STYLE,
        className="mb-4",
    )


def create_layout():
    category_options = [{"label": category, "value": category} for category in CATEGORIES]
    document_type_options = [{"label": doc_type, "value": doc_type} for doc_type in DOCUMENT_TYPES]

    year_marks = {
        YEAR_MIN: str(YEAR_MIN),
        YEAR_SLIDER_MAX: str(YEAR_MAX),
    }

    return html.Div(
        [
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div(
                                        "IT 2012 · Lab 13",
                                        className="text-uppercase fw-bold",
                                        style={"color": "#38bdf8", "letterSpacing": "0.12rem"},
                                    ),
                                    html.H1(
                                        "News Media Monitoring Dashboard",
                                        className="fw-bold mt-2 mb-2",
                                        style={"fontSize": "2.6rem"},
                                    ),
                                    html.P(
                                        "Interactive dashboard for monitoring integrated news, scraped, document, OCR, audio, and video records.",
                                        className="lead mb-0",
                                        style={"color": "#cbd5e1"},
                                    ),
                                ],
                                lg=8,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                "Live Status",
                                                className="text-uppercase small fw-semibold",
                                                style={"color": "#94a3b8"},
                                            ),
                                            html.H4(
                                                "Dashboard Online",
                                                className="fw-bold mb-1",
                                                style={"color": "#22c55e"},
                                            ),
                                            html.Div(
                                                "Auto-refreshing ticker enabled",
                                                style={"color": "#cbd5e1"},
                                            ),
                                        ]
                                    ),
                                    style=CARD_STYLE,
                                ),
                                lg=4,
                                className="mt-4 mt-lg-0",
                            ),
                        ],
                        align="center",
                        className="py-5",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(metric_card("Records", f"{METRICS['total_records']:,}", "Filtered dataset size", "kpi-total-records"), md=6, lg=2),
                            dbc.Col(metric_card("Categories", METRICS["category_count"], "Unique categories", "kpi-category-count"), md=6, lg=2),
                            dbc.Col(metric_card("Document Types", METRICS["document_type_count"], "Integrated source types", "kpi-document-type-count"), md=6, lg=2),
                            dbc.Col(metric_card("Avg Rating", METRICS["avg_rating"], "Mean rating score", "kpi-avg-rating"), md=6, lg=2),
                            dbc.Col(metric_card("Avg Popularity", METRICS["avg_popularity"], "Mean popularity", "kpi-avg-popularity"), md=6, lg=2),
                            dbc.Col(metric_card("Avg Length", METRICS["avg_content_length"], "Mean content length", "kpi-avg-content-length"), md=6, lg=2),
                        ],
                        className="g-3 mb-4",
                    ),
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("Category", className="fw-semibold mb-2"),
                                                dcc.Dropdown(
                                                    id="category-filter",
                                                    options=category_options,
                                                    value=[],
                                                    multi=True,
                                                    placeholder="Select categories",
                                                    style={"color": "#111827"},
                                                ),
                                            ],
                                            lg=3,
                                            md=6,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Document Type", className="fw-semibold mb-2"),
                                                dcc.Dropdown(
                                                    id="document-type-filter",
                                                    options=document_type_options,
                                                    value=[],
                                                    multi=True,
                                                    placeholder="Select document types",
                                                    style={"color": "#111827"},
                                                ),
                                            ],
                                            lg=3,
                                            md=6,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Published Year Range", className="fw-semibold mb-2"),
                                                dcc.RangeSlider(
                                                    id="year-range-filter",
                                                    min=YEAR_MIN,
                                                    max=YEAR_SLIDER_MAX,
                                                    step=1,
                                                    value=[YEAR_MIN, YEAR_MAX],
                                                    marks=year_marks,
                                                    tooltip={"placement": "bottom", "always_visible": False},
                                                    allowCross=False,
                                                ),
                                            ],
                                            lg=3,
                                            md=6,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Search", className="fw-semibold mb-2"),
                                                dbc.Input(
                                                    id="search-input",
                                                    type="text",
                                                    placeholder="Search title, category, content...",
                                                    debounce=True,
                                                ),
                                            ],
                                            lg=3,
                                            md=6,
                                        ),
                                    ],
                                    className="g-3",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dbc.Button(
                                                    "Reset Filters",
                                                    id="reset-filters-btn",
                                                    color="info",
                                                    outline=True,
                                                    className="mt-3",
                                                ),
                                                html.Span(
                                                    id="filter-summary",
                                                    className="ms-3",
                                                    style={"color": "#cbd5e1"},
                                                ),
                                            ],
                                            width=12,
                                        )
                                    ]
                                ),
                            ]
                        ),
                        style={
                            "background": "#0f172a",
                            "border": "1px solid #334155",
                            "borderRadius": "18px",
                        },
                        className="mb-4",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                graph_card(
                                    "Top News Records by Popularity",
                                    "top-records-chart",
                                    "Ranks records by popularity and rating score.",
                                ),
                                lg=6,
                            ),
                            dbc.Col(
                                graph_card(
                                    "Rating Score Distribution",
                                    "rating-distribution-chart",
                                    "Shows how rating scores are distributed in the filtered dataset.",
                                ),
                                lg=6,
                            ),
                        ],
                        className="g-4",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                graph_card(
                                    "Popularity vs Content Length",
                                    "popularity-content-chart",
                                    "Scatter plot for relationship between record length and popularity.",
                                ),
                                lg=6,
                            ),
                            dbc.Col(
                                graph_card(
                                    "Records Over Time",
                                    "year-trend-chart",
                                    "Yearly record volume with average rating.",
                                ),
                                lg=6,
                            ),
                        ],
                        className="g-4",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                graph_card(
                                    "Document Type Breakdown",
                                    "document-type-chart",
                                    "Compares integrated source/document types.",
                                ),
                                lg=6,
                            ),
                            dbc.Col(
                                graph_card(
                                    "Live News Monitoring Ticker",
                                    "live-ticker-chart",
                                    "Simulated live stream updated with dcc.Interval.",
                                ),
                                lg=6,
                            ),
                        ],
                        className="g-4",
                    ),
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Top Matching Records", className="fw-bold mb-3"),
                                html.Div(id="top-records-table"),
                            ]
                        ),
                        style=GRAPH_CARD_STYLE,
                        className="mb-4",
                    ),
                    dcc.Interval(
                        id="live-interval",
                        interval=3000,
                        n_intervals=0,
                    ),
                    dcc.Store(id="live-data-store", data=[]),
                ],
                fluid=True,
                style={"maxWidth": "1500px"},
            )
        ],
        style=PAGE_STYLE,
    )