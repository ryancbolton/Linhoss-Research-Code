import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
#note will need scipy for the df.coeff "kendall" method to work

df = pd.read_csv('MasterDoplhinData_Input.csv')

print(df.isnull().sum()) # prints number of data gaps for each column (station)

def NRWC_method():
    headers = df.head(0) # makes a list of all the station names
    headers2 = df.head(0) # makes another list of all the station names
    columns = [] #initializes list for every column or station of corrcoeffs
    weights = [] #initializes list of list for weights
    sumrowweights = []
    missingvals = []

    for header in headers:
        column = [] # initializes list for each column of corrcoeffs for each station
        for head in headers2:
            a = df[header].corr(df[head], method = 'pearson') # calculates each corrcoeff
            #adds each correlation coefficient for a given station to the "column" list
            column.append(a)

        columns.append(column) #creates a list of lists (each column of coefficients)
        # n = len(column)
    
    # For every station column
    for column in columns:
        weight = []
        # For every correlation coeffecient between each station for each station
        for coeff in column:
            try:
                w = (len(df.index) - 2)*(coeff**2)/(1-coeff**2) # weight equation
                
                # Filters out weights calculated where corrcoeff = 1
                if w > 1000000: 
                    # print('#')
                    weight.append(0)
                else:
                    # print(str(w))
                    weight.append(w)

            except ZeroDivisionError:
                print('#') 
        weights.append(weight)
        # print("############################################")


    #Gets the sum of the weights for each station
    sumweights = np.add.reduceat(weights, np.arange(0, len(weights), len(df.columns)))
    sumweights = sumweights[0] #reduces list of list 

    #Replace missing (NaN) values by column (axis=1)
    df.fillna(value=0, method=None, axis=1, inplace=True) #inplace=true modifies the df in realtime instead of making a copy with inplace=false

    # For every list of weights for each station
    for weight in weights:
        # For every row in the data frame
        for index, row in df.iterrows():
            # Multiply each value in each row by each corresponding weight and sum them
            a = sum(x * y for x, y in zip(row, weight))
            sumrowweights.append(a)

    # For each sum of the weights in the sumweights list
    for sumweight in sumweights:
        missingval = []
        # For every set (12 total) of sum of rows x weights (4018 rows x weight 1, 4018 rows x weight 2, etc)
        for sumrowweight in sumrowweights:
            # Missing value = (sum of weights x values)/sum of weights for each station 
            b = sumrowweight/sumweight
            missingval.append(b)
        missingvals.append(missingval)

    j = 0
    final_df = pd.DataFrame()
    for col in df:
        for i, row in df.iterrows():
            if df[col][i] == 0:
                df[col][i] = missingvals[j][i]
        final_df = final_df.append(df)
        j = j + 1

    final_df.to_csv('Replaced_values.csv', encoding='utf-8', index=False)

            
def plot():
    corrMatrix = df.corr() # turns dataframe into correlation matrix      
    ax = sn.heatmap(corrMatrix, annot=True, cbar_kws={'label': 'Correlation Coefficient'}, xticklabels=['A','B','E','M','N','O', 'P', 'Q', 'R', 'S', 'T', 'U'], yticklabels=['A','B','E','M','N','O', 'P', 'Q', 'R', 'S', 'T', 'U']) # turns corr matrix into heatmap representation
    ax.figure.axes[-1].yaxis.label.set_size(15) # change font size of colorbar label
    sn.set(font_scale=1.2) # sets size of x and y labels and data values

    plt.show()

NRWC_method()
plot()