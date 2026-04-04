import os
import numpy as np
import pandas as pd
from lxml import etree as ET


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

    columns = ["date_to", "PM10", "PM2.5"]

    for sifra in sifra_vals:
        postaja_elements = tree.xpath(f'//postaja[@sifra="{sifra}"]')

        new_rows = []

        for postaja in postaja_elements:
            date_to = postaja.find('datum_do').text if postaja.find('datum_do') is not None else np.nan
            pm10 = postaja.find('pm10').text if postaja.find('pm10') is not None else np.nan
            pm2_5 = postaja.find('pm2.5').text if postaja.find('pm2.5') is not None else np.nan

            new_rows.append([date_to, pm10, pm2_5])

        new_df = pd.DataFrame(new_rows, columns=columns)

        new_df = new_df.replace("", np.nan)
        new_df = new_df.replace("<1", 1)
        new_df = new_df.replace("<2", 2)

        new_df["date_to"] = pd.to_datetime(new_df["date_to"], errors="coerce")
        new_df["PM10"] = pd.to_numeric(new_df["PM10"], errors="coerce")
        new_df["PM2.5"] = pd.to_numeric(new_df["PM2.5"], errors="coerce")

        file_path = os.path.join(output_dir, f"{sifra}.csv")

        if os.path.exists(file_path):
            old_df = pd.read_csv(file_path)

            old_df["date_to"] = pd.to_datetime(old_df["date_to"], errors="coerce")
            old_df["PM10"] = pd.to_numeric(old_df["PM10"], errors="coerce")
            old_df["PM2.5"] = pd.to_numeric(old_df["PM2.5"], errors="coerce")

            df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            df = new_df

        before_dedup = len(df)

        df = df.drop_duplicates(subset=["date_to"], keep="last")
        df = df.sort_values(by="date_to")

        after_dedup = len(df)

        df.to_csv(file_path, index=False)

        print(f"Updated: {sifra}.csv | rows before dedup: {before_dedup}, after dedup: {after_dedup}")


if __name__ == "__main__":
    preprocess_air_data()