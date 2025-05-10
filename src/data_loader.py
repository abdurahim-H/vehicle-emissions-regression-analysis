import pandas as pd
import requests
from io import StringIO

def load_2014_data():
    """Load the 2014 vehicle emissions dataset"""
    url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
    df = pd.read_csv(url)
    cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
    return cdf

def load_2025_data():
    """Load the 2025 vehicle emissions dataset"""
    file_id = "1hkSHZBQ_C6NaWuLcH0VbwDt_jCdK4wzu"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    response = requests.get(url)
    content = response.content.decode('latin-1')
    df = pd.read_csv(StringIO(content))
    cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
    return cdf