import os
import sys
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Dash

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.dashboard.layout import create_layout
from src.dashboard.callbacks import register_callbacks


app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="News Media Monitoring Dashboard",
    suppress_callback_exceptions=True,
)

server = app.server

app.layout = create_layout()

register_callbacks(app)


if __name__ == "__main__":
    app.run(
        debug=os.environ.get("DASH_DEBUG", "true").lower() == "true",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 8050)),
    )