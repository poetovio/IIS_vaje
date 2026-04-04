from pathlib import Path
import requests
from datetime import datetime


def fetch_air_data():
    try:
        url = "https://www.arso.gov.si/xml/zrak/ones_zrak_urni_podatki_7dni.xml"

        response = requests.get(url)
        response.raise_for_status()

        project_root = Path(__file__).resolve().parents[2]
        raw_dir = project_root / "data" / "raw" / "air"
        raw_dir.mkdir(parents=True, exist_ok=True)

        file_path = raw_dir / "air_data.xml"
        with open(file_path, "wb") as file:
            file.write(response.content)

        print(f"Fetching successful. Data saved to {file_path} at {datetime.now()}")

    except requests.RequestException as e:
        print(f"Error fetching data: {e}")


if __name__ == "__main__":
    fetch_air_data()