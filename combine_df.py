import os
import glob
import pandas as pd

def combine_pickles(input_dir, output_file):
    # Recursively find all pickle files in the input directory
    pkl_files = glob.glob(os.path.join(input_dir, '**', '*.pkl'), recursive=True)
    print(f"Found {len(pkl_files)} pickle files in '{input_dir}'.")
    
    # Load each pickle file into a list
    df_list = []
    for file in pkl_files:
        try:
            df = pd.read_pickle(file)
            df_list.append(df)
            print(f"Loaded: {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    # Concatenate all DataFrames (each is one row) into one big DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)
    print("Combined DataFrame shape:", combined_df.shape)
    
    # Save the combined DataFrame to the output file
    combined_df.to_pickle(output_file)
    print(f"Combined DataFrame saved to '{output_file}'.")
    return combined_df

if __name__ == '__main__':
    # Specify the directory where your individual pickles are stored.
    input_directory = "C:/Users/Ethier Lab/Documents/GitHub/Decoder-Processing/DataSET/Jango_ISO_2015"
    # Specify the path for the combined pickle file.
    output_pickle = input_directory + "/combined_1.pkl"
    
    combined_df = combine_pickles(input_directory, output_pickle)
