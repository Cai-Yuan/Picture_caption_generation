          scatter_kws = {'alpha': 0.8, 's': 60}, fit_reg = False,
          size = 12, aspect = 1.2);

# Plot labeling
plt.xlabel("Site EUI", size = 28)
plt.ylabel('Cardiac Function Score', size = 28)
plt.title('Density Plot of Cardiac Function Scor by ', size = 36);
figsize(6, 5)
# Extract the building types
features['Largest Property Use Type'] = data.dropna(subset = ['score'])['Largest Property Use Type']

# Limit to building types with more than 100 observations (from previous code)
features = features[features['Largest Property Use Type'].isin(types)]

# Use seaborn to plot a scatterplot of Score vs Log Source EUI
sns.lmplot('Chronic Blood pressure (Average)', 'score', 
          hue = 'Largest Property Use Type', data = features,
          scatter_kws = {'alpha': 0.8, 's': 60}, fit_reg = False,
          size = 12, aspect = 1.2);

# Plot labeling
plt.xlabel("Site EUI", size = 28)
plt.ylabel('Cardiac Function Score', size = 28)
plt.title('Density Plot of Cardiac Function Scor by ', size = 36);
figsize(6, 5)
# Extract the building types
features['Largest Property Use Type'] = data.dropna(subset = ['score'])['Largest Property Use Type']

# Limit to building types with more than 100 observations (from previous code)
features = features[features['Largest Property Use Type'].isin(types)]

# Use seaborn to plot a scatterplot of Score vs Log Source EUI
sns.lmplot('Chronic Blood pressure (Average)', 'score', 
          hue = 'Largest Property Use Type', data = features,
          scatter_kws = {'alpha': 0.8, 's': 60}, fit_reg = False,
          size = 12, aspect = 1.2);

# Plot labeling
plt.xlabel("Chronic Blood pressure (Average)", size = 28)
plt.ylabel('Cardiac Function Score', size = 28)
plt.title('Density Plot of Cardiac Function Scor by ', size = 36);
12*12
12**12
runfile('C:/Users/cyuan/Desktop/数据结构的编程/untitled1.py', wdir='C:/Users/cyuan/Desktop/数据结构的编程')

## ---(Mon Dec  9 22:58:36 2019)---
runfile('C:/Users/cyuan/Desktop/数据结构的编程/untitled1.py', wdir='C:/Users/cyuan/Desktop/数据结构的编程')

## ---(Tue Dec 10 10:06:22 2019)---
runfile('C:/Users/cyuan/Desktop/数据结构的编程/untitled1.py', wdir='C:/Users/cyuan/Desktop/数据结构的编程')

## ---(Tue Dec 10 13:24:48 2019)---
def dir_file_count(directory):
    return sum([len(files) for r, d, files in os.walk(directory)])
dir_file_count(r'C:\Users\cyuan\Desktop\PIC_DATA')
import sys
import os
import argparse
import random
import time
import datetime
dir_file_count(r'C:\Users\cyuan\Desktop\PIC_DATA')
directory=r'C:\Users\cyuan\Desktop\PIC_DATA'
sum([len(r) for r, d, files in os.walk(directory)])
sum([len(d) for r, d, files in os.walk(directory)])
sum([r for r, d, files in os.walk(directory)])
def name_correct(name):
    return re.sub(r'[^a-zA-Z,:]', ' ', name).title()
    
import re
name_correct('ansduifhue4455746')
os.listdir(directory)
def subdirectory_file_count(master_directory):
    subdirectories = os.listdir(master_directory) 
    subdirectory_count = len(subdirectories)

    subdirectory_names = []
    subdirectory_file_counts = []

    for subdirectory in subdirectories:
        current_directory = os.path.join(master_directory, subdirectory)
        file_count = len(os.listdir(current_directory))
        subdirectory_names.append(subdirectory)
        subdirectory_file_counts.append(file_count)
    
    return subdirectory_names, subdirectory_file_counts
    
subdirectory_file_count(directory)
rescale = 1./255

dirs = os.listdir(directory)
for i in range(len(dirs)):
    print(i, dirs[i])
    
dir_name = r"data/output/models/"
cur_dir =dir_name+dirs[0]+"/"
cur_dir =dir_name+dirs[0]+"/"
model_names = os.listdir(cur_dir)
for i in range(len(model_names)):
    print(i, model_names[i])
    