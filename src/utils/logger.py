import logging
import os

LOG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "pipeline.log"))

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)