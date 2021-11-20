import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
#note will need scipy for the df.coeff "kendall" method to work

df = pd.read_csv('Mean_SpecificConductance.csv')

print(df.isnull().sum()) # prints number of data gaps for each column (station)


def corrcoeff():
    headers = df.head(0) # makes a list of all the station names
    headers2 = df.head(0) # makes another list of all the station names
    count = 0

    for header in headers:
        print(header)
        for head in headers2:
            print('************************ PEARSON  ************************')

            print(df[header].corr(df[head], method = 'pearson'))
            #print('************************ KENDALL  ************************')
            #print(df[header].corr(df[head], method = 'kendall'))
            #print('************************ SPEARMAN ************************')
            #print(df[header].corr(df[head], method = 'spearman'))
            print(header)
            print(head)

        print('##############################################################') 
        print(count+1)
        count = count + 1


def weighted_avg():
    headers = df.head(0) # makes a list of all the station names
    headers2 = df.head(0) # makes another list of all the station names
    totals = [] # initialize list for the total of all the corrcoeffs for each column (station)
    avgs = [] # initializes list for the average of each total in totals list
    columns = [] #initializes list for every column or station of corrcoeffs
    weights = [] #initializes list for weights

    for header in headers:
        column = [] # initializes list for each column of corrcoeffs for each station
        sum1 = 0
        avg1 = 0
        for head in headers2:
            a = df[header].corr(df[head], method = 'pearson') # each corrcoeff

            #adds each correlation coefficient for a given station to the "column" list
            column.append(a)

        columns.append(column) #creates a list of lists (each column of coefficients)

        #after each column is complete, takes the sum and adds it to the "totals" list
        sum1 = sum(column)
        # n = len(column)
        avg1 = sum1/len(column) # calculates average of corrcoeffs
        totals.append(sum1)
        avgs.append(avg1) # adds averages to avg list
    
    for column in columns:
        for coeff in column:
            try:
                w = (4018 - 2)*(coeff**2)/(1-coeff**2) # weight equation
                
                if w > 1000000:
                    print('#')
                    weights.append(0)
                else:
                    print(str(w))
                    weights.append(w)
                    # print(str(coeff))

            except ZeroDivisionError:
                print('#')


                
        print("############################################")


    # weightedavgs = []
    # for avg in avgs:
    #     for header in headers:
    #         weightsum = 0
    #         weigthedavg = 0
    #         weighted_val = df[header]*avg
    #         weightsum = weightsum + weighted_val
    #         print(weighted_val)
    #         print(weightsum)
    #     weightedavg = weighted


    print(totals)
    print(avgs)
    print(weights)

# def interpolate():
#     headers = df.head(0) # makes a list of all the station names
#     headers2 = df.head(0) # makes another list of all the station names
#     count = 0

#     for header in headers:
#         df[header].interpolate(method='linear', direction = 'forward', inplace=True) 
#         # print(df[header])
#         print(df.isnull().sum()) # prints number of data gaps for each column (station)
#         for head in headers2:
#             df[head].interpolate(method='linear', direction = 'forward', inplace=True) 
#             print('************************ PEARSON  ************************')
#             print(df[header].corr(df[head], method = 'pearson'))

#         print('##############################################################') 
#         print(count+1)
#         count = count + 1


def plot():
    corrMatrix = df.corr() # turns dataframe into correlation matrix      
    ax = sn.heatmap(corrMatrix, annot=True, cbar_kws={'label': 'Correlation Coefficient'}, xticklabels=['A','B','E','M','N','O', 'P', 'Q', 'R', 'S', 'T', 'U'], yticklabels=['A','B','E','M','N','O', 'P', 'Q', 'R', 'S', 'T', 'U']) # turns corr matrix into heatmap representation
    ax.figure.axes[-1].yaxis.label.set_size(15) # change font size of colorbar label
    sn.set(font_scale=1.2) # sets size of x and y labels and data values

    plt.show()


# corrcoeff()
weighted_avg()
# interpolate()
plot()



















import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
#note will need scipy for the df.coeff "kendall" method to work

df = pd.read_csv('Mean_SpecificConductance.csv')

print(df.isnull().sum()) # prints number of data gaps for each column (station)


def corrcoeff():
    headers = df.head(0) # makes a list of all the station names
    headers2 = df.head(0) # makes another list of all the station names
    count = 0

    for header in headers:
        print(header)
        for head in headers2:
            print('************************ PEARSON  ************************')

            print(df[header].corr(df[head], method = 'pearson'))
            #print('************************ KENDALL  ************************')
            #print(df[header].corr(df[head], method = 'kendall'))
            #print('************************ SPEARMAN ************************')
            #print(df[header].corr(df[head], method = 'spearman'))
            print(header)
            print(head)

        print('##############################################################') 
        print(count+1)
        count = count + 1


def weighted_avg():
    headers = df.head(0) # makes a list of all the station names
    headers2 = df.head(0) # makes another list of all the station names
    totals = [] # initialize list for the total of all the corrcoeffs for each column (station)
    avgs = [] # initializes list for the average of each total in totals list
    columns = [] #initializes list for every column or station of corrcoeffs
    weights = [] #initializes list of list for weights
    stations = ['A', 'B', 'E', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U']
    sumrowweights = []
    missingvals = []

    for header in headers:
        column = [] # initializes list for each column of corrcoeffs for each station
        sum1 = 0
        avg1 = 0
        for head in headers2:
            a = df[header].corr(df[head], method = 'pearson') # each corrcoeff

            #adds each correlation coefficient for a given station to the "column" list
            column.append(a)

        columns.append(column) #creates a list of lists (each column of coefficients)

        #after each column is complete, takes the sum and adds it to the "totals" list
        sum1 = sum(column)
        # n = len(column)
        avg1 = sum1/len(column) # calculates average of corrcoeffs
        totals.append(sum1)
        avgs.append(avg1) # adds averages to avg list
    
    i = 0
    # For every station column
    for column in columns:
        weight = []
        # For every correlation coeffecient between each station for each station
        for coeff in column:
            try:
                w = (4018 - 2)*(coeff**2)/(1-coeff**2) # weight equation
                
                if w > 1000000:
                    print('#')
                    weight.append(0)
                else:
                    print(str(w))
                    weight.append(w)
                    # print(str(coeff))

            except ZeroDivisionError:
                print('#') 
        weights.append(weight)
        print(stations[i])
        i = i + 1
        print("############################################")

    

    print(totals)
    print(avgs)
    print(weights)

    #Gets the sum of the weights for each station
    sumweights = np.add.reduceat(weights, np.arange(0, len(weights), 12))
    sumweights = sumweights[0] #reduces list of list 
    print(sumweights)

    #Replace missing values by column (axis=1)
    df.fillna(value=0, method=None, axis=1, inplace=True)

    #Prints the values for each column (station)
    # i = 0
    # for header in headers:
    #     print(stations[i])
    #     for i, row in df.iterrows():
    #         print(row[header])  
    #     print("############################################")
    #     i = i + 1

    print(df)
    # For every row in the data frame
    for index, row in df.iterrows():
        # For every list of weights for each station
        for weight in weights:
            # Multiply each value in each row by each corresponding weight and sum them
            a = sum(x * y for x, y in zip(row, weight))
            #print(a)
            sumrowweights.append(a)
    # For each sum of the weights in the sumweights list
    for sumweight in sumweights:
        for i in range(1,len(sumrowweights)):
            # Missing value = (sum of weights x values)/sum of weights for each station 
            b = a/sumweight
            #print('####################################')
            #print(b)

            # if df[i][index] == 0:
            #     df[i][index] = b
            
            # if pd.notnull():
            #     df[i][index] = b

    for col in df:
        if df[col][index] == 0:
            df[col][index] = b
    print(df)


            


def plot():
    corrMatrix = df.corr() # turns dataframe into correlation matrix      
    ax = sn.heatmap(corrMatrix, annot=True, cbar_kws={'label': 'Correlation Coefficient'}, xticklabels=['A','B','E','M','N','O', 'P', 'Q', 'R', 'S', 'T', 'U'], yticklabels=['A','B','E','M','N','O', 'P', 'Q', 'R', 'S', 'T', 'U']) # turns corr matrix into heatmap representation
    ax.figure.axes[-1].yaxis.label.set_size(15) # change font size of colorbar label
    sn.set(font_scale=1.2) # sets size of x and y labels and data values

    plt.show()


# corrcoeff()
weighted_avg()
plot()







import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
#note will need scipy for the df.coeff "kendall" method to work

df = pd.read_csv('Mean_SpecificConductance.csv')

print(df.isnull().sum()) # prints number of data gaps for each column (station)


def corrcoeff():
    headers = df.head(0) # makes a list of all the station names
    headers2 = df.head(0) # makes another list of all the station names
    count = 0

    for header in headers:
        print(header)
        for head in headers2:
            print('************************ PEARSON  ************************')

            print(df[header].corr(df[head], method = 'pearson'))
            #print('************************ KENDALL  ************************')
            #print(df[header].corr(df[head], method = 'kendall'))
            #print('************************ SPEARMAN ************************')
            #print(df[header].corr(df[head], method = 'spearman'))
            print(header)
            print(head)

        print('##############################################################') 
        print(count+1)
        count = count + 1


def NRWC_method():
    headers = df.head(0) # makes a list of all the station names
    headers2 = df.head(0) # makes another list of all the station names
    totals = [] # initialize list for the total of all the corrcoeffs for each column (station)
    avgs = [] # initializes list for the average of each total in totals list
    columns = [] #initializes list for every column or station of corrcoeffs
    weights = [] #initializes list of list for weights
    stations = ['A', 'B', 'E', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U']
    sumrowweights = []
    missingvals = []

    for header in headers:
        column = [] # initializes list for each column of corrcoeffs for each station
        sum1 = 0
        avg1 = 0
        for head in headers2:
            a = df[header].corr(df[head], method = 'pearson') # each corrcoeff

            #adds each correlation coefficient for a given station to the "column" list
            column.append(a)

        columns.append(column) #creates a list of lists (each column of coefficients)

        #after each column is complete, takes the sum and adds it to the "totals" list
        sum1 = sum(column)
        # n = len(column)
        avg1 = sum1/len(column) # calculates average of corrcoeffs
        totals.append(sum1)
        avgs.append(avg1) # adds averages to avg list
    
    i = 0
    # For every station column
    for column in columns:
        weight = []
        # For every correlation coeffecient between each station for each station
        for coeff in column:
            try:
                w = (4018 - 2)*(coeff**2)/(1-coeff**2) # weight equation
                
                if w > 1000000:
                    print('#')
                    weight.append(0)
                else:
                    print(str(w))
                    weight.append(w)
                    # print(str(coeff))

            except ZeroDivisionError:
                print('#') 
        weights.append(weight)
        print(stations[i])
        i = i + 1
        print("############################################")


    print(totals)
    print(avgs)
    print(weights)

    #Gets the sum of the weights for each station
    sumweights = np.add.reduceat(weights, np.arange(0, len(weights), 12))
    sumweights = sumweights[0] #reduces list of list 
    print(sumweights)

    #Replace missing (NaN) values by column (axis=1)
    df.fillna(value=0, method=None, axis=1, inplace=True)

    print(df)
    # For every list of weights for each station
    for weight in weights:
        # For every row in the data frame
        for index, row in df.iterrows():
            # Multiply each value in each row by each corresponding weight and sum them
            a = sum(x * y for x, y in zip(row, weight))
            #print(a)
            sumrowweights.append(a)

    # For each sum of the weights in the sumweights list
    for sumweight in sumweights:
        missingval = []
        # For every set (12 total) of sum of rows x weights (4018 rows x weight 1, 4018 rows x weight 2, etc)
        for sumrowweight in sumrowweights:
            # Missing value = (sum of weights x values)/sum of weights for each station 
            b = sumrowweight/sumweight
            #print('####################################')
            #print(b)
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

    # j = 0
    # final_df = pd.DataFrame()
    # for i in range(len(df)):
    #     if df.iloc[i,0] == 0:
    #         df.iloc[i,0] = missingvals[j][i]
    # final_df = final_df.append(df)
    # j = j + 1

    print(sumrowweights)
    print(missingvals)
    # print(df)
    final_df.to_csv('Replaced_values.csv', encoding='utf-8', index=False)

            


def plot():
    corrMatrix = df.corr() # turns dataframe into correlation matrix      
    ax = sn.heatmap(corrMatrix, annot=True, cbar_kws={'label': 'Correlation Coefficient'}, xticklabels=['A','B','E','M','N','O', 'P', 'Q', 'R', 'S', 'T', 'U'], yticklabels=['A','B','E','M','N','O', 'P', 'Q', 'R', 'S', 'T', 'U']) # turns corr matrix into heatmap representation
    ax.figure.axes[-1].yaxis.label.set_size(15) # change font size of colorbar label
    sn.set(font_scale=1.2) # sets size of x and y labels and data values

    plt.show()


# corrcoeff()
NRWC_method()
plot()






import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
#note will need scipy for the df.coeff "kendall" method to work

df = pd.read_csv('MasterDoplhinData_Input.csv')

print(df.isnull().sum()) # prints number of data gaps for each column (station)


def corrcoeff():
    headers = df.head(0) # makes a list of all the station names
    headers2 = df.head(0) # makes another list of all the station names
    count = 0

    for header in headers:
        print(header)
        for head in headers2:
            print('************************ PEARSON  ************************')

            print(df[header].corr(df[head], method = 'pearson'))
            #print('************************ KENDALL  ************************')
            #print(df[header].corr(df[head], method = 'kendall'))
            #print('************************ SPEARMAN ************************')
            #print(df[header].corr(df[head], method = 'spearman'))
            print(header)
            print(head)

        print('##############################################################') 
        print(count+1)
        count = count + 1


def NRWC_method():
    headers = df.head(0) # makes a list of all the station names
    headers2 = df.head(0) # makes another list of all the station names
    totals = [] # initialize list for the total of all the corrcoeffs for each column (station)
    avgs = [] # initializes list for the average of each total in totals list
    columns = [] #initializes list for every column or station of corrcoeffs
    weights = [] #initializes list of list for weights
    stations = ['A', 'B', 'E', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U']
    sumrowweights = []
    missingvals = []

    for header in headers:
        column = [] # initializes list for each column of corrcoeffs for each station
        sum1 = 0
        avg1 = 0
        for head in headers2:
            a = df[header].corr(df[head], method = 'pearson') # each corrcoeff
            #adds each correlation coefficient for a given station to the "column" list
            column.append(a)

        columns.append(column) #creates a list of lists (each column of coefficients)

        #after each column is complete, takes the sum and adds it to the "totals" list
        sum1 = sum(column)
        # n = len(column)
        avg1 = sum1/len(column) # calculates average of corrcoeffs
        totals.append(sum1)
        avgs.append(avg1) # adds averages to avg list
    
    i = 0
    # For every station column
    for column in columns:
        weight = []
        # For every correlation coeffecient between each station for each station
        for coeff in column:
            try:
                w = (len(df.index) - 2)*(coeff**2)/(1-coeff**2) # weight equation
                
                # Filters out weights calculated where corrcoeff = 1
                if w > 1000000: 
                    print('#')
                    weight.append(0)
                else:
                    print(str(w))
                    weight.append(w)
                    # print(str(coeff))

            except ZeroDivisionError:
                print('#') 
        weights.append(weight)
        print(stations[i])
        i = i + 1
        print("############################################")


    # print(totals)
    # print(avgs)
    print(weights)

    #Gets the sum of the weights for each station
    sumweights = np.add.reduceat(weights, np.arange(0, len(weights), len(df.columns)))
    sumweights = sumweights[0] #reduces list of list 
    print(sumweights)

    #Replace missing (NaN) values by column (axis=1)
    df.fillna(value=0, method=None, axis=1, inplace=True) #inplace=true modifies the df in realtime instead of making a copy with inplace=false

    # print(df)
    # For every list of weights for each station
    for weight in weights:
        # For every row in the data frame
        for index, row in df.iterrows():
            # Multiply each value in each row by each corresponding weight and sum them
            a = sum(x * y for x, y in zip(row, weight))
            #print(a)
            sumrowweights.append(a)

    # For each sum of the weights in the sumweights list
    for sumweight in sumweights:
        missingval = []
        # For every set (12 total) of sum of rows x weights (4018 rows x weight 1, 4018 rows x weight 2, etc)
        for sumrowweight in sumrowweights:
            # Missing value = (sum of weights x values)/sum of weights for each station 
            b = sumrowweight/sumweight
            #print('####################################')
            #print(b)
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

    # j = 0
    # final_df = pd.DataFrame()
    # for i in range(len(df)):
    #     if df.iloc[i,0] == 0:
    #         df.iloc[i,0] = missingvals[j][i]
    # final_df = final_df.append(df)
    # j = j + 1

    # print(sumrowweights)
    # print(missingvals)
    # print(df)
    final_df.to_csv('Replaced_values.csv', encoding='utf-8', index=False)

            
def plot():
    corrMatrix = df.corr() # turns dataframe into correlation matrix      
    ax = sn.heatmap(corrMatrix, annot=True, cbar_kws={'label': 'Correlation Coefficient'}, xticklabels=['A','B','E','M','N','O', 'P', 'Q', 'R', 'S', 'T', 'U'], yticklabels=['A','B','E','M','N','O', 'P', 'Q', 'R', 'S', 'T', 'U']) # turns corr matrix into heatmap representation
    ax.figure.axes[-1].yaxis.label.set_size(15) # change font size of colorbar label
    sn.set(font_scale=1.2) # sets size of x and y labels and data values

    plt.show()


# corrcoeff()
NRWC_method()
plot()