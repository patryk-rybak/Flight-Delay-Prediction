import pandas as pd
from datetime import datetime
import requests
import json
import threading
import concurrent.futures

# Lock for thread-safe operations
lock = threading.Lock()

def timeconvert(year, month, day, time_as_int):
    # Function to convert time in int format to a formatted string
    time_as_str = str(time_as_int).zfill(4)
    formatted_time = f"{time_as_str[:2]}:{time_as_str[2:]}"
    date_to_convert = datetime(year, month, day)
    start = f"{date_to_convert.strftime('%Y-%m-%d')}T{formatted_time}:00"
    return start

def mapping(val, whattomap):
    # Function to map values using a predefined mapping from a JSON file
    with open('mappings.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        data_dict = data.get(whattomap)
        resp_dict = data_dict.get(str(val))
    return resp_dict

def collectingdata(grid, rok, orig, dataname, df):
    # Function to fetch weather data and populate DataFrame
    print("Fetching data")
    with open('pass.txt', 'r') as file:
        API_key = file.read().strip()

    origin_name = mapping(orig, "ORIGIN")
    city_name = origin_name["City"]
    state = origin_name["State"]
    URL = f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city_name},{state}/{rok}-01-01/{rok}-12-31?key={API_key}'

    with lock:
        response = requests.get(URL)

    weather_day_data = response.json()

    for day in range(numofdaysinyear):

        try:
            desired_day_data = weather_day_data["days"][day]
            moon = weather_day_data["days"][day].get('moonphase', None)
        except Exception as e:
            print(f"Error: {e}. Missing data for {day}. Going to the next iteration.")
            continue

        squer = grid[day]

        if len(squer) > 1:
            for hours in squer[1:]:
                index_row = hours[0]
                hour = hours[1]
                hour = hour // 100

                try:
                    desired_hour_data = desired_day_data["hours"][hour]
                except KeyError as e:
                    print(f"Error: {e}. Missing data for {day} and {hour}. Going to the next iteration.")
                    continue
                except Exception as ex:
                    print(f"Error: {ex}. Going to the next iteration.")
                    continue

                important_data = {
                    'MOONPHASE': moon,
                    'CONDITIONS': desired_hour_data.get('conditions', None),
                    'CLOUDCOVER': desired_hour_data.get('cloudcover', None),
                    'VISIBILITY': desired_hour_data.get('visibility', None),
                    'PRESSURE': desired_hour_data.get('pressure', None),
                    'WINDDIR': desired_hour_data.get('winddir', None),
                    'WINDSPEED': desired_hour_data.get('windspeed', None),
                    'WINDGUST': desired_hour_data.get('windgust', None),
                    'SNOWDEPTH': desired_hour_data.get('snowdepth', None),
                    'SNOW': desired_hour_data.get('snow', None),
                    'PRECIPPROB': desired_hour_data.get('precipprob', None),
                    'PRECIP': desired_hour_data.get('precip', None),
                    'DEW': desired_hour_data.get('dew', None),
                    'HUMIDITY': desired_hour_data.get('humidity', None),
                    'TEMP': desired_hour_data.get('temp', None),
                    'FEELSLIKE': desired_hour_data.get('feelslike', None)
                }

                for key, value in important_data.items():
                    df.at[index_row, key] = value

                # Handling 'preciptype' which may return an array
                value = desired_hour_data.get('preciptype', None)
                if value:
                    for c in value:
                        df.at[index_row, c] = 1

    print(orig)
    df.to_csv(dataname)

def daycode(year, month, day, baseyear):
    # Function to calculate the day code based on the difference between two dates
    formatted_date = datetime(year, month, day)
    base_date = datetime(baseyear, 1, 1)
    difference = formatted_date - base_date
    return difference.days

def gridmaker(baseyear, df):
    # Function to create a grid for mapping data based on DataFrame
    print("Creating grid")
    grid = [[] for _ in range(numofdaysinyear)]

    for index, row in df.iterrows():
        year = int(row.iloc[8])
        month = int(row.iloc[9])
        day = int(row.iloc[10])
        nrofday = daycode(year, month, day, baseyear)

        deptime = int(row.iloc[3])

        if not grid[nrofday]:
            grid[nrofday].append((year, month, day))

        grid[nrofday].append((index, deptime))

    print("Grid created")
    return grid

def starttreat(i, year):
    # Function to start the data processing for each origin
    filename = f'originsplited_2017\df_2017_{i}.csv'

    df = pd.read_csv(filename, dtype={'CONDITIONS': str})
    df = df.set_index('ID')

    collectingdata(gridmaker(year, df), year, i, filename, df)

def main():
    # Main function to initiate data processing in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
        futures = {executor.submit(starttreat, start, 2017): start for start in range(numoforigins)}

        for future in concurrent.futures.as_completed(futures):
            start = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error in process {start}: {e}")

# Set the number of origins and days in a year
numoforigins = 359
numofdaysinyear = 365

if __name__ == "__main__":
    main()
