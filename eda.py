import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Null Eater
# Used for: internal data cleaning in the following functions
#
# This function takes two arrays, x and y, and returns those arrays without any values that are null in either x or y.
#
# Example:
# x = [0, 1, None, 2, 3]
# y = ['a', None, 'b', 'c', 'd']
#
# x' = [0, 2, 3]
# y' = ['a','c','d']

def nullEater(x,y):
    if(len(x) != len(y)):
        raise IndexError('nullEater: length of arrays x and y must be equal')
    xmask = pd.isnull(x)
    ymask = pd.isnull(y)
    x = x[np.logical_or(xmask,ymask) == False]
    y = y[np.logical_or(xmask,ymask) == False]
    return x, y

# Layered Correlation Histogram
# Used for: Classification tasks with quantitative features.
# 
# This function takes two arrays - an explanatory variable (predictor) and a response variable (target).
# It will then generate a histogram showing the distribution of the target variable grouped by the target variable,
# for the purposes of exploratory data analysis. Output graphs are stored in the visualizations/exploration/ folder
# of the project.
# 
# target: Response variable column.
# 
# predictor: Explanatory variable column.
# 
# predictor/target/target_values_mask: Optional names for predictor and response in chart labels. Default is to match
#   predictor and target column names exactly or to just use the values in target.
# 
# style: PLT style to use. Defaults to default style.
# 
# dpi: DPI of image to save. Default 160.
# 
# Fontsize: Font of all text in the graph. Default 10.

def corrHist(predictor,target,predictor_mask='',target_mask='',target_values_mask='',style='default',dpi=160,fontsize=10):
    if not predictor_mask:
        predictor_mask = predictor.name
    if not target_mask:
        target_mask = target.name
    if target_values_mask:
        target = np.asarray([target_values_mask[i] for i in target])
    plt.style.use(style)
    predictor, target = nullEater(predictor,target)
    plt.figure(figsize=(10,8))
    for i in np.unique(target):
        plt.hist(predictor[target == i],10,alpha=0.5,label=i)
    plt.legend(loc='upper right')
    plt.xlabel(predictor_mask)
    plt.ylabel('Frequency')
    plt.title('Histogram of {} grouped by {}'.format(predictor_mask,target_mask))
    plt.xticks(rotation=90)
    plt.savefig('visualizations/exploration/hist{}by{}.png'.format(predictor_mask,target_mask),dpi=dpi,bbox_inches='tight')

# Stacked Percentage Bar Chart
# Used for: Classification tasks with quantitative and qualitative features.
# 
# This function takes two arrays: a response variable and an explanatory variable.
# It will then generate a normalized stacked bar plot showing the percentage distribution of the target
# variable within the predictor variable for the purposes of exploratory data analysis.
# If the predictor is quantitative, this function will automatically separate it into ten bins before
# proceeding.
# 
# Arguments:
#
# target: Response variable.
# 
# predictor: Explanatory variable.
# 
# predictor/target_mask: Optional names for predictor and response in chart labels. Default is to match
#   predictor and target column names exactly.
# 
# predictor/target_values_mask: Accepts a dictionary in the format:
# 
#     {value1: new_name1, value2: new_name2, ...}
# 
#   For example, in the Titanic dataset:
#   
#     {0: 'Perished', 1: 'Survived'}
# 
# style: PLT style to use. Defaults to default style.
# 
# verbose: Adds number labels to bar slices. Default true.
# 
# dpi: DPI of image to save. Default 160.
# 
# Fontsize: Font of all text in the graph. Default 10.

def percBar(predictor,target,predictor_mask='',target_mask='',predictor_values_mask='',target_values_mask='',style='default',verbose=True,dpi=160,fontsize=10):
    tname = target.name
    pname = predictor.name
    if not predictor_mask:
        predictor_mask = predictor.name
    if not target_mask:
        target_mask = target.name
    if predictor_values_mask:
        predictor = np.asarray([predictor_values_mask[i] for i in predictor])
    if target_values_mask:
        target = np.asarray([target_values_mask[i] for i in target])
    if(str(predictor[0]).isnumeric()): # This step must be performed after the label masks are applied; otherwise, if, for example, a bool were stored as 0 and 1, this piece of code would execute.
        predictor = pd.cut(predictor,10,labels=False)
    predictor, target = nullEater(predictor,target)
    plt.style.use(style)
    plt.rc('font',size=fontsize)
    lengths = []
    labels = np.append(np.unique(predictor),tname)
    explan = np.unique(predictor)
    response = np.unique(target)
    if(len(labels) > 25):
        print('Error: Target variable ' + str(tname) + ' contains more than 25 variants. Terminating procedure.')
        return
    plt.figure(figsize=(1.25*len(labels),6))
    for j in range(0,len(response)):
        for i in range(0,len(explan)):
            # For each variant in the target variable:
            # For each variant in the predictor variable, save the number of times that the target variant and predictor variant appear together.
            try: # np.unique with return_counts=True returns an array with the format [[False, True],[number of False,number of True]]
                lengths = np.append(lengths,np.unique(np.logical_and(predictor == explan[i],target==response[j]),return_counts=True)[1][1]/np.unique(predictor == explan[i],return_counts=True)[1][1])
            except IndexError: # Catches cases where p(x, y) is 0 or 1.
                if(np.unique(np.logical_and(predictor == explan[i],target==response[j]))):
                    lengths.append(1)
                else:
                    lengths.append(0)
        # At the end, append the number of the target variant divided by the total number of target instances.
        # This gives us the native distribution of the target variable, which allows us to compare how different the various
        # predictor categories are.
        lengths = np.append(lengths,np.unique(target == response[j],return_counts=True)[1][1]/len(target))
    n = np.asarray(np.unique(predictor,return_counts=True)[1])  # Store the count of each prediction category, in case the user wants them displayed.
    n = np.append(n,len(target))
    for k in range(0,len(response)):
        if(k==0):
            previousData=[0]*len(labels)
        else:
            previousData=[0]*len(labels)
            for l in range(k,0,-1):
                # This step is necessary in order to properly place the bar charts
                # Essentially, we are printing the bar chart one layer at a time; bottom layer, next layer, and so on
                # until we reach the top. In order to place the top layers correctly, we need to know where the last layer
                # ended.
                previousData=np.add(lengths[(l-1)*len(labels):l*len(labels)],previousData)
        plt.bar(np.where(labels==tname,target_mask,labels),lengths[k*len(labels):(k+1)*len(labels)],bottom=previousData,label=response[k])
        if verbose:
            labels = np.where(labels==tname,target_mask,labels)
            for m in range(0,len(labels)):
                plt.text(labels[m],np.add(lengths[k*len(labels):(k+1)*len(labels)],previousData)[m],round(100*lengths[k*len(labels):(k+1)*len(labels)][m],2),ha="center",va="top")
                if(k==len(response)-1):
                    plt.text(labels[m],np.add(lengths[k*len(labels):(k+1)*len(labels)],previousData)[m],"N=" + str(n[m]),ha="center",va="bottom")
    plt.legend(bbox_to_anchor=(1.01,1.01),loc='upper left')
    plt.xlabel(predictor_mask)
    plt.ylabel('Distribution')
    if verbose:
        plt.title('Distribution of {} within {}'.format(target_mask,predictor_mask),pad=15.0)
    else:
        plt.title('Distribution of {} within {}'.format(target_mask,predictor_mask))
    plt.xticks(rotation=90)
    plt.savefig('visualizations/exploration/perc{}by{}.png'.format(pname,tname),dpi=dpi,bbox_inches='tight')

# Mutual Information
# Used for: Correlation between categorical variables

# This function takes two lists or arrays, then spits out the mutual information between the two variables contained therein.
# Mutual information is a symmetric measure of association. It indicates how much the uncertainty of each variable is reduced by knowledge of the other.

def mutualInformation(x,y,axis=0):
    x,y = nullEater(x,y)
    explan = np.unique(x)
    response = np.unique(y)
    px = list(map(lambda i: i/len(x),np.unique(x,return_counts=True)[1]))
    py = list(map(lambda i: i/len(y),np.unique(y,return_counts=True)[1]))
    pxy = pd.DataFrame(columns=range(0,len(response)),index=range(0,len(explan)))
    for i in range(0,len(response)):
        for j in range(0,len(explan)):
            # In the following dataframe, [i][j] is the probability that condition j and condition i are both true.
            # i and j correspond both response/explan and py/px, respectively.
            try: # np.unique with return_counts=True returns an array with the format [[False, True],[number of False,number of True]]
                pxy[i][j] = np.unique(np.logical_and(x == explan[j],y==response[i]),return_counts=True)[1][1]/len(x)
            except IndexError: # Catches cases where p(x, y) is 0 or 1.
                if(np.unique(np.logical_and(x == explan[j],y==response[i]))):
                    pxy[i][j] = 1
                else:
                    pxy[i][j] = 0
    mi = 0 # Container for mutual information.
    for k in range(0,len(response)):
        for l in range(0,len(explan)):
            try:
                mi += float(pxy[k][l]*math.log2(pxy[k][l]/(px[l]*py[k])))
            except ValueError: # Catches cases where p(x, y) is 0. The convention is to treat 0log0 as 0, since xlogx approaches 0 asymptotically as x approaches 0.
                continue
    return mi

