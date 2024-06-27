import json
from datetime import datetime, timedelta

def check_json_structure(data):
    if "days" in data:
        for day in data["days"]:
            if "hours" in day:
                hours_expected = 24
                unique_hours = set()
                for hour in day["hours"]:
                    if "datetime" in hour:
                        hour_str = hour["datetime"]
                        try:
                            hour_datetime = datetime.strptime(hour_str, "%H:%M:%S")
                            if hour_datetime in unique_hours:
                                print(f"Powtórzona godzina {hour_str} w dniu {day['datetime']}.")
                            else:
                                unique_hours.add(hour_datetime)
                        except ValueError:
                            print(f"Błąd w formacie godziny dla {hour_str} w dniu {day['datetime']}.")
                if len(unique_hours) != hours_expected:
                    print(f"Brak którejś z godzin w dniu {day['datetime']}.")
            else:
                print(f"Brak informacji o godzinach w dniu {day['datetime']}.")

def check_month_structure(data):
    if "days" in data:
        dates_in_month = set()

        for day in data["days"]:
            if "datetime" in day:
                date_str = day["datetime"]
                try:
                    date = datetime.strptime(date_str, "%Y-%m-%d")
                    dates_in_month.add(date)
                except ValueError:
                    print(f"Błąd w formacie daty dla dnia {date_str}.")

        expected_dates = set()
        current_date = datetime.strptime(data["days"][0]["datetime"], "%Y-%m-%d")

        for _ in range(len(data["days"])):
            expected_dates.add(current_date)
            current_date += timedelta(days=1)

        missing_dates = expected_dates - dates_in_month

        if missing_dates:
            print(f"Brakuje danych dla dni: {', '.join(str(date.date()) for date in missing_dates)}")

with open('temp.json') as f:
    json_data = json.load(f)

check_json_structure(json_data)
check_month_structure(json_data)
