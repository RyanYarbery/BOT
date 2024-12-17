import os
import csv
import logging
import requests
import time
from datetime import datetime, timezone, timedelta

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
API_URL = "https://api.binance.us/api/v3/klines"
INTERVAL = "1m"
HEADER = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume']
DIRECTORY = "CSVs"
SYMBOLS = ["ETHUSDT", "BTCUSDT"]

# Ensure the directory exists
if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)

def clean_csv(file_path):
    temp_file_path = file_path + '.temp'
    
    with open(file_path, 'rb') as f:
        data = f.read()
    
    cleaned_data = data.replace(b'\x00', b'')
    
    with open(temp_file_path, 'wb') as f:
        f.write(cleaned_data)
    
    return temp_file_path

def get_last_fetch_time_from_csv(filename):
    cleaned_file_path = clean_csv(filename)
    
    try:
        with open(cleaned_file_path, 'r', newline='') as file:
            last_row = None
            for last_row in csv.reader(file): pass
            if last_row and last_row[0] != 'Open Time':
                return datetime.strptime(last_row[0], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    except FileNotFoundError:
        logging.info(f"CSV file not found: {filename}.")
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
    finally:
        os.remove(cleaned_file_path)  # Clean up the temporary file
    return None

def calculate_limit(last_fetch_time):
    if last_fetch_time is None:
        logging.debug("No last fetch time found, setting limit to 999.")
        return 999
    elapsed_time = datetime.now(timezone.utc) - last_fetch_time
    elapsed_minutes = elapsed_time.total_seconds() / 60
    calculated_limit = min(int(elapsed_minutes), 999)
    logging.debug(f"Elapsed time since last fetch: {elapsed_minutes} minutes, setting limit to: {calculated_limit}")
    return calculated_limit

def fetch_klines(limit, symbol):
    params = {
        'symbol': symbol,
        'interval': INTERVAL,
        'limit': limit
    }
    max_retries = 5
    retry_interval = 10

    for attempt in range(max_retries):
        try:
            response = requests.get(API_URL, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                logging.error(f"Failed to fetch klines: {response.text}")
                time.sleep(retry_interval)
        except requests.RequestException as e:
            logging.error(f"Exception during API request: {e}")
            time.sleep(retry_interval)
    return []

def log_to_csv(filename, data):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(HEADER)
        writer.writerow(data)

def get_most_recent_csv_file(symbol):
    csv_files = [f for f in os.listdir(DIRECTORY) if f.endswith('.csv') and symbol in f]
    if not csv_files:
        return None
    csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(DIRECTORY, x)), reverse=True)
    return os.path.join(DIRECTORY, csv_files[0])

def get_last_fetch_time(symbol):
    most_recent_filename = get_most_recent_csv_file(symbol)
    
    if most_recent_filename:
        last_fetch_time = get_last_fetch_time_from_csv(most_recent_filename)
        return last_fetch_time
    else:
        logging.info(f"No CSV files found for symbol: {symbol}.")
        return None

def get_filename(symbol, date):
    return f'{DIRECTORY}/{date.strftime("%Y-%m")}_{symbol}Klines.csv'

def main():
    while True:
        start_time = time.time()
        cycle_time = 60  # Time in seconds for each cycle
        
        for symbol in SYMBOLS:
            last_fetch_time = get_last_fetch_time(symbol)
            limit = calculate_limit(last_fetch_time)
            logging.info(f"Fetching klines for {symbol} with limit: {limit}")

            klines = fetch_klines(limit, symbol)
            if klines:
                for kline in klines[:-1]:  # Process all but the most recent to ensure completeness
                    kline_time = datetime.fromtimestamp(kline[0] / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                    kline_datetime = datetime.fromtimestamp(kline[0] / 1000, tz=timezone.utc)
                    current_month_filename = get_filename(symbol, kline_datetime)
                    data_to_log = [kline_time] + kline[1:6]
                    log_to_csv(current_month_filename, data_to_log)
                    logging.info(f"Logged kline for {symbol}: {data_to_log}")

        # After the initial fetch, always set limit to 2 for subsequent runs
        limit = 2

        end_time = time.time()
        elapsed_time = end_time - start_time
        sleep_time = max(0, cycle_time - elapsed_time)
        logging.info(f"Cycle completed in {elapsed_time} seconds. Sleeping for {sleep_time} seconds.")
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()
