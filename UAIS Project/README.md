NOTICE: Datasets are not available for public use. 

This program is designed to perform several data analysis tasks using the k-nearest neighbors (kNN) algorithm and handle missing data using imputation techniques. Here's a summary of what the program does:

1. **Data Retrieval and Preprocessing:**
   - It retrieves data from an Excel file named 'OFS_summary_gyemark.xls', specifically from the 'Site Leadership' sheet, focusing on columns 'Status 1', 'Status 2', and 'Level'. 
   - It preprocesses the data by converting categorical variables into numerical format using one-hot encoding.

2. **Data Imputation:**
   - It utilizes kNN imputation to fill in missing values in the 'Level' column of the dataset.
   - The imputed data is then saved to an Excel file named 'OFS_summary_gyemark.xlsx'.

3. **kNN Algorithm:**
   - It applies the kNN algorithm to the dataset for classification tasks.
   - It splits the dataset into training and testing sets, encodes labels, trains the kNN classifier, and makes predictions.
   - Predicted labels are appended to the testing data and saved to the same Excel file.

4. **Data Visualization:**
   - It generates bar plots to visualize the frequency of safe and unsafe occurrences for 'Status 1' and 'Status 2'.
   - It compares the safety of 'Status 1' and 'Status 2' through bar plots.
   - It creates bar plots to display the frequency of accidents in different areas and buildings, respectively.

5. **Main Function:**
   - It orchestrates the execution of all the defined functions.

Overall, the program aims to analyze safety-related data, impute missing values, perform classification tasks using kNN, and visualize the results through various plots for better understanding and decision-making.
