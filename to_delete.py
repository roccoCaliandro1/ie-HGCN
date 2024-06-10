import pandas as pd

file_name = 'spatial_network_train'
dimension = 'big'

input_file = './OneDrive/preprocess/twitter/cleaned/' + dimension + '/graph/' + file_name + '.txt'

# Define column names explicitly
column_names = ['follow_user_id', 'followed_user_id', 'weight']

# Load the DataFrame with specified column names
df = pd.read_csv(input_file, sep='\t', comment='#', names=column_names)

# Filter rows where the 'weight' column (third column) is greater than or equal to 0.5
filtered_df = df[df['weight'] >= 0.5]

# Define the output file path
out_file = './OneDrive/preprocess/twitter/cleaned/' + dimension + '/graph/' + file_name + '_new.txt'

# Save the filtered DataFrame to a new text file
filtered_df.to_csv(out_file, sep='\t', index=False)
