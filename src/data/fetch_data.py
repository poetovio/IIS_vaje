import requests
from datetime import datetime
import xml.etree.ElementTree as ET

def fetch_air_data():
    try:
        url = "https://www.arso.gov.si/xml/zrak/ones_zrak_urni_podatki_7dni.xml"

        response = requests.get(url)
        response.raise_for_status()

        file_path = "data/raw/air/air_data.xml"
        with open(file_path, "wb") as file:
            file.write(response.content)

        print(f"Fetching successful. Data saved to {file_path} at {datetime.now()}")

    except requests.RequestException as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    fetch_air_data()