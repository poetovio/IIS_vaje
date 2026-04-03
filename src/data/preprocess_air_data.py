import numpy as np
import pandas as pd
from lxml import etree as ET
import os

def preprocess_air_data():
    output_dir = "data/preprocessed/air"
    os.makedirs(output_dir, exist_ok=True)

    with open("data/raw/air/air_data.xml", "rb") as file:
        tree = ET.parse(file)
        root = tree.getroot()

    print(f"Version: {root.attrib['verzija']}")
    print(f"Source: {root.find('vir').text}")
    print(f"Suggested Capture: {root.find('predlagan_zajem').text}")
    print(f"Suggested Capture Period: {root.find('predlagan_zajem_perioda').text}")
    print(f"Preparation Date: {root.find('datum_priprave').text}")

    sifra_vals = set(tree.xpath('//postaja/@sifra'))

    for sifra in sifra_vals:
        postaja_elements = tree.xpath(f'//postaja[@sifra="{sifra}"]')

        data = []

        for postaja in postaja_elements:
            date_to = postaja.findtext('datum_do')
            pm10 = postaja.findtext('pm10')
            pm2_5 = postaja.findtext('pm2.5')

            data.append([date_to, pm10, pm2_5])

        new_df = pd.DataFrame(data, columns=["date_to", "PM10", "PM2.5"])

        new_df = new_df.replace("", np.nan)
        new_df = new_df.replace("<1", 1)
        new_df = new_df.replace("<2", 2)

        new_df["date_to"] = pd.to_datetime(new_df["date_to"], errors="coerce")
        new_df["PM10"] = pd.to_numeric(new_df["PM10"], errors="coerce")
        new_df["PM2.5"] = pd.to_numeric(new_df["PM2.5"], errors="coerce")

        file_path = f"{output_dir}/{sifra}.csv"

        if os.path.exists(file_path):
            old_df = pd.read_csv(file_path)
            old_df["date_to"] = pd.to_datetime(old_df["date_to"], errors="coerce")
            df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            df = new_df

        df = df.drop_duplicates(subset=["date_to"])
        df = df.sort_values(by="date_to")

        df.to_csv(file_path, index=False)

        print(f"Updated: {sifra}.csv")

if __name__ == "__main__":
    preprocess_air_data()