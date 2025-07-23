#!/usr/bin/env python3
import sys
import os
sys.stdout.flush()

print("=== DEBUG START ===")
sys.stdout.flush()

# Set environment
os.environ['HF_HOME'] = '/pasteur/appa/scratch/mcampi/COSI2/.cache'
os.environ['TRANSFORMERS_CACHE'] = '/pasteur/appa/scratch/mcampi/COSI2/.cache'

print("Environment set")
sys.stdout.flush()

try:
    print("Importing libraries...")
    sys.stdout.flush()
    import torch
    import pandas as pd
    from transformers import CamembertTokenizer, CamembertForSequenceClassification
    
    print("Libraries imported successfully")
    sys.stdout.flush()
    
    print("Testing Excel file...")
    sys.stdout.flush()
    df = pd.read_excel('data/COSI_EVOLUTION_by_need.xlsx', engine='openpyxl')
    print(f"Excel loaded: {df.shape}")
    sys.stdout.flush()
    
    print("All tests passed!")
    sys.stdout.flush()
    
except Exception as e:
    print(f"ERROR: {e}")
    sys.stdout.flush()
