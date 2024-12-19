import pandas as pd

# Load the CSV file
csv_file_path = '/home/npatel23/gokhale_user/Crop Classification Project/HLS_CDL_Data/california_chip_data.csv'  # Replace with the path to your CSV file
df = pd.read_csv(csv_file_path)

# Filter data for the training set (2014 to 2016)
training_df = df[df['year'].isin([2014, 2015, 2016])]['chip']

# Filter data for the testing set (2016 and 2017)
testing_df = df[df['year'].isin([2017, 2018])]['chip']

# Define paths for the output files
training_file_path = '/home/npatel23/gokhale_user/Crop Classification Project/HLS_CDL_Data/training_california_chips.csv'  # Replace with your desired path
testing_file_path = '/home/npatel23/gokhale_user/Crop Classification Project/HLS_CDL_Data/testing_california_chips.csv'  # Replace with your desired path

# Save the training and testing chip ids to CSV
training_df.to_csv(training_file_path, index=False)
testing_df.to_csv(testing_file_path, index=False)

print(f"Training chip ids have been saved to {training_file_path}")
print(f"Testing chip ids have been saved to {testing_file_path}")
