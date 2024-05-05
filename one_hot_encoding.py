import pandas as pd
import numpy as np

# Define the column names for the table and read in the dataset.
column_names = ['name', 'year', 'selling_price', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']
df = pd.read_csv("chosen.csv", names=column_names, header=None)

# Transmission: One-hot encoding for the categorial data. Manual = 1, Automatic = 0.
df['transmission'] = (df['transmission'] != 'automatic').astype(int) 

# Fuel: one-hot encoding by creating a column, by comparing the fuel, getting the boolean value, and converting a binary 0, 1.
df['Diesel'] = (df['fuel'] == 'Diesel').astype(int) 
df['Petrol'] = (df['fuel'] == 'Petrol').astype(int)
df['CNG'] = (df['fuel'] == 'CNG').astype(int) 
df['LPG'] = (df['fuel'] == 'LPG').astype(int)
df['electric'] = (df['fuel'] == 'electric').astype(int)

# Seller type: one-hot encoding.
df['seller_individual'] = (df['seller_type'] == 'Individual').astype(int) 
df['seller_dealer'] = (df['seller_type'] == 'Dealer').astype(int) 
df['seller_trustmark'] = (df['seller_type'] == 'Trustmark Dealer').astype(int) 

# Owner: one-hot encoding.
df['first_owner'] = (df['owner'] == 'First Owner').astype(int) 
df['second_owner'] = (df['owner'] == 'Second Owner').astype(int) 
df['third_owner'] = (df['owner'] == 'Third Owner').astype(int) 
df['fourth_owner'] = (df['owner'] == 'Fourth & Above Owner').astype(int) 

# Remove the fuel, owner, etc.
df = df.drop(['fuel', 'owner', 'seller_type'], axis=1)

# Convert the selling price to american USD from rupee.
df['selling_price'] /= 83

# Save the modified DataFrame to a new CSV file, which is used for all our implementations.
df.to_csv("modified_chosen.csv", index=False)
