from pathlib import Path
from datetime import datetime

import requests
import yaml


def fetch_air_data():
    project_root = Path(__file__).resolve().parents[2]

    with open(project_root / "params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)["fetch"]

    url = params["url"]

    raw_dir = project_root / "data" / "raw" / "air"
    raw_dir.mkdir(parents=True, exist_ok=True)

    file_path = raw_dir / "air_data.xml"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(file_path, "wb") as file:
            file.write(response.content)

        print(f"Fetching successful. Data saved to {file_path} at {datetime.now()}")

    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        raise


if __name__ == "__main__":
    fetch_air_data()