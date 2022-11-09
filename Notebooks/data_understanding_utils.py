# Functions to help with data understanding 

import pandas as pd
import plotly
import matplotlib
import scipy
import numpy
from matplotlib import pyplot
import seaborn as sns
def read_data(path, file_name):
    """
    Read data from a csv file
    :param path: path to the file
    :param file_name: name of the file
    :return: data frame
    """
    return pd.read_csv(path+file_name , sep=';')


# with default parameters
def info_data(data,option,group_type):
    """
    Summarize the data
    :param data: data frame
    :return: summary of the data
    """
    if(option == 'shape'):
        print(data.shape, end='\n\n')
    elif (option == 'head'):
        print(data.head(), end='\n\n')
    elif (option == 'tail'):
        print(data.tail(), end='\n\n')
    elif (option == 'describe'):
        print(data.describe(), end='\n\n')
    elif (option == 'info'):
        print(data.info(), end='\n\n')
    elif option == group_type:
        print(data.groupby(group_type).size(), end='\n\n')
    elif(option == 'isnull'):
        print("Number of null values: \n",data.isnull().sum(), end='\n\n')
    elif(option == 'probabilitie'):
        len(data[data[group_type]])


def prob_value_on_table(data, data_name: str, collumn:str, value):
    len(data[data[collumn] == value]) / len(data)

def data_summarization(dataset, option):
    if option == "hist":
        dataset.hist()
        pyplot.show()
    elif(option == 'box'):
        dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
        pyplot.show()
    elif(option == 'density'):
        dataset.plot(kind='density', subplots=True, layout=(2,2), sharex=False, sharey=False)
        pyplot.show()
    elif(option == 'correlation'):
        pyplot.figure(figsize=(10, 10))
        sns.heatmap(dataset.corr(), annot=True, cbar=True, cmap='coolwarm')
    elif(option == 'skew'):
        print(dataset.skew())
    elif(option == 'kurtosis'):
        print(dataset.kurtosis())
    elif(option == 'describe'):
        print(dataset.describe())
    elif(option == 'scatter_matrix'):
        pd.plotting.scatter_matrix(dataset)
        pyplot.show()
    

def data_visualization(data,type_graph):
    """
    Visualize the data
    :param data: data frame
    :return: visualization of the data
    """
    return data.plot()

def check_duplicates(data, data_name: str,collumns):
    """
    Check for duplicates
    :param data: data frame
    :param data_name: name of the data table
    :param collumns: collumns to check for duplicates
    :return: number of duplicates
    """
    count_duplicated =  data[data.duplicated(collumns)]
    result = count_duplicated.value_counts().value_counts()
    if(result.empty):
        print("No duplicates found in the data")
        return
    print(f"Duplicated rows on {data_name}: \n{count_duplicated}")
    print(f"Number of repeated values in {data_name} : {result}")
    return count_duplicated.value_counts().value_counts()
