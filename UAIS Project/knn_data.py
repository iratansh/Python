"""
This program utilizes kNN to impute missing data and analyzes data from various datasets
Author: Ishaan Ratanshi
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt

def get_data_from_OFS():
    """
    Access data from excel file
    Returns data
    """
    file = pd.read_excel('OFS_summary_gyemark.xls', sheet_name="Site Leadership", usecols=['Status 1', 'Status 2', 'Level'])
    data = pd.get_dummies(file.iloc[:,[0, 1]])
    labels = file.iloc[:, [2]]
    return data, labels

def impute_data(data):
    """
    Impute missing data in 'Level' Column
    """
    imputer = KNNImputer(n_neighbors=3)
    imputed_data = imputer.fit_transform(data)
    imputed_data_df = pd.DataFrame(imputed_data)
    imputed_data_df.to_excel('OFS_summary_gyemark.xlsx', index=False, header=True)
    return imputed_data_df

def kNN(data, labels):
    """
    kNN algorithm
    """
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, random_state=1)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train.values.ravel())
    knn = KNeighborsClassifier(n_neighbors=30)
    knn.fit(x_train, y_train_encoded)
    y_pred = knn.predict(x_test)
    x_test_with_labels = x_test.copy()
    x_test_with_labels['Predicted Labels'] = label_encoder.inverse_transform(y_pred)
    x_test_with_labels.to_excel('OFS_summary_gyemark.xlsx', index=False, header=True)

def bar_plot_for_safety():
    """
    Create bar plot for saftey frequencies
    """
    file = pd.read_excel('OFS_summary_gyemark.xlsx', sheet_name="Sheet1", usecols=['Status 1_Safe', 'Status 1_Unsafe', 'Status 2_Safe', 'Status 2_Unsafe'])
    safe_counts = file['Status 1_Safe'].value_counts()
    unsafe_counts = file['Status 1_Unsafe'].value_counts()
    safe2_counts = file['Status 2_Safe'].value_counts()
    unsafe2_counts = file['Status 2_Unsafe'].value_counts()
    plt.figure(figsize=(10, 6))

    # Plot for Status 1
    plt.subplot(2, 2, 1)
    safe_counts.plot(kind='bar', color='blue', alpha=0.7)
    plt.title('Status 1 Safe Counts')
    plt.subplot(2, 2, 2)
    unsafe_counts.plot(kind='bar', color='red', alpha=0.7)
    plt.title('Status 1 Unsafe Counts')

    # Plot for Status 2
    plt.subplot(2, 2, 3)
    safe2_counts.plot(kind='bar', color='green', alpha=0.7)
    plt.title('Status 2 Safe Counts')
    plt.subplot(2, 2, 4)
    unsafe2_counts.plot(kind='bar', color='orange', alpha=0.7)
    plt.title('Status 2 Unsafe Counts')

    plt.tight_layout()
    plt.show()

def compare_safety():
    """
    Create bar plots comparing Safety of status 1 and status 2 
    """
    file = pd.read_excel('OFS_summary_gyemark.xlsx', sheet_name="Sheet1", usecols=['Status 1_Safe', 'Status 1_Unsafe', 'Status 2_Safe', 'Status 2_Unsafe'])
    status1_safe_counts = file['Status 1_Safe'].value_counts()
    status1_unsafe_counts = file['Status 1_Unsafe'].value_counts()
    status2_safe_counts = file['Status 2_Safe'].value_counts()
    status2_unsafe_counts = file['Status 2_Unsafe'].value_counts()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Plot for Status 1
    status1_counts = pd.concat([status1_safe_counts, status1_unsafe_counts], axis=1)
    status1_counts.columns = ['Safe', 'Unsafe']
    status1_counts.plot(kind='bar', ax=axes[0], color=['blue', 'red'], alpha=0.7)
    axes[0].set_title('Status 1 Safe vs Unsafe')
    axes[0].set_ylabel('Frequency')
    axes[0].set_xlabel('Status')
    axes[0].set_xticklabels(['True', 'False'], rotation=0)
    axes[0].legend(loc='upper center')
    # Plot for Status 2
    status2_counts = pd.concat([status2_safe_counts, status2_unsafe_counts], axis=1)
    status2_counts.columns = ['Safe', 'Unsafe']
    status2_counts.plot(kind='bar', ax=axes[1], color=['green', 'orange'], alpha=0.7)
    axes[1].set_title('Status 2 Safe vs Unsafe')
    axes[1].set_ylabel('Frequency')
    axes[1].set_xlabel('Status')
    axes[1].set_xticklabels(['True', 'False'], rotation=0)
    axes[1].legend(loc='upper center')

    plt.tight_layout()
    plt.show()

def display_areas_of_concern():
    """
    Compare areas of concern
    """
    file = pd.read_excel('OFS_summary_gyemark.xls', sheet_name="Site Leadership", usecols=['Area 1', 'Area 2'])
    all_areas = pd.concat([file['Area 1'], file['Area 2']], ignore_index=True)

    # Calculate the frequency of each area
    area_counts = all_areas.value_counts()

    # Plot the frequency of accidents in each area
    plt.figure(figsize=(8, 6))
    area_counts.plot(kind='bar', color='skyblue', alpha=0.7)
    plt.title('Accident Frequency in Different Areas')
    plt.xlabel('Area')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def display_buildings_of_concern():
    """
    Compare buildings of concern
    """
    file = pd.read_excel('OFS_summary_gyemark.xls', sheet_name="Site Leadership", usecols=['Building 1', 'Building 2'])
    all_buildings = pd.concat([file['Building 1'], file['Building 2']], ignore_index=True)
    # Calculate the frequency of each building
    building_counts = all_buildings.value_counts()

    # Plot the frequency of accidents in each building
    plt.figure(figsize=(8, 6))
    building_counts.plot(kind='bar', color='blue', alpha=0.7)
    plt.title('Accident Frequency in Different Buildings')
    plt.xlabel('Building')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    """
    Main Function
    """
    data, labels = get_data_from_OFS()
    impute_data(labels)
    kNN(data, labels)
    bar_plot_for_safety()
    compare_safety()
    display_areas_of_concern()
    display_buildings_of_concern()
    
if __name__ == '__main__':
    main()