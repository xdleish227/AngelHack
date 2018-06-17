import pickle
import pandas as pd
import argparse
import csv
import sys
from pprint import pprint
import dateutil.parser

# Create airport name -> codeSEQ mapping
name_to_code = csv.DictReader(open("id_code.csv", 'r'))
                            
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
depart = name_to_code[args.departurreAirportFS]
arrive = name_to_code[args.arrivalAirportFS]

tmp = date.date()
tmp = str(tmp).split('-')

year = tmp[0]
month = tmp[1].lstrip('0')
day = tmp[2].lstrip('0')
time = date.time()
m = 0
m = m + time.hour*60
m = m + time.minute

d = {'ORIGIN_AIRPORT_ID':depart, 'DEST_AIRPORT_ID':arrive, 'MONTH':month, 'DAY_OF_MONTH':day, 'CRS_DEP_TIME':m}

x = pd.DataFrame([d], columns=['ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'MONTH', 'DAY_OF_MONTH', 'CRS_DEP_TIME'])

# Prediction
y_pred = model.predict_proba(x)
print(type(y_pred))
