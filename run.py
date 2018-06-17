import pickle
import pandas as pd
import argparse
import csv
import sys
from pprint import pprint
import dateutil.parser

# Create airport name -> codeSEQ mapping
reader = csv.DictReader(open("airportcode.csv", 'r'))
dict_list = []
for line in reader:
    dict_list.append(line)

name_to_code = {}
for i in dict_list:
    short_name= i['Description'].split(":")[-1].strip()
    code = i['Code']
    name_to_code[short_name] = code

# Take arguments
parser = argparse.ArgumentParser()
parser.add_argument('--departurreAirportFS')
parser.add_argument('--arrivalAirportFS')
parser.add_argument('--departureTime')
args = parser.parse_args()

date= dateutil.parser.parse(args.departureTime)

# Load model
model_pkl = open("LogRegression_Model.pkl", "rb")
model = pickle.load(model_pkl)

# Data preparation
depart = args.departurreAirportFS
arrive = args.arrivalAirportFS

tmp = date.date()
tmp = str(tmp).split('-')

year = tmp[0]
month = tmp[1].lstrip('0')
day = tmp[2].lstrip('0')
time = date.time()
m = 0
m = m + time.hour*60
m = m + time.minute

d = {'ORIGIN_AIRPORT_SEQ_ID':depart, 'DEST_AIRPORT_SEQ_ID':arrive, 'MONTH':month, 'DAY_OF_MONTH':day, 'CRS_DEP_TIME':m}

x = pd.DataFrame([d], columns=['ORIGIN_AIRPORT_SEQ_ID', 'DEST_AIRPORT_SEQ_ID', 'MONTH', 'DAY_OF_MONTH', 'CRS_DEP_TIME'])

# Prediction
y_pred = model.predict_proba(x)
print(y_pred)
