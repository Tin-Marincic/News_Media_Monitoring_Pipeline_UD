import json
import logging
from pathlib import Path
def setup_logging(log_file):
   logging.basicConfig(
       filename=log_file,
       level=logging.INFO,
       format="%(asctime)s - %(levelname)s - %(message)s"
   )

def read_json(file_path):
   try:
       with open(file_path, "r", encoding="utf-8") as f:
           data = json.load(f)
       logging.info(f"Successfully read JSON file: {file_path}")
       return data
   except FileNotFoundError:
       logging.error(f"File not found: {file_path}")
   except json.JSONDecodeError:
       logging.error(f"Invalid JSON format in file: {file_path}")


def read_text(file_path):
   try:
       with open(file_path, "r", encoding="utf-8") as f:
           text = f.read()
       logging.info(f"Successfully read text file: {file_path}")
       return text
   except FileNotFoundError:
       logging.error(f"File not found: {file_path}")
