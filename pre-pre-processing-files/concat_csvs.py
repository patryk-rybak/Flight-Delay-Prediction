import os
import pandas as pd

directory_path = "/home/meks/Desktop/data/"

df = pd.DataFrame()

for year in os.listdir(directory_path):
    year_path = os.path.join(directory_path, year)

    if os.path.isdir(year_path):
        for file in os.listdir(year_path):
            file_path = os.path.join(year_path, file)

            day_data = pd.read_csv(file_path)
            df = pd.concat([df, day_data], ignore_index=True)
            print(f"{file}")

print(df.info())
print(df.describe())
print(df)

df.to_csv(f"complete_data.csv", index=False)
