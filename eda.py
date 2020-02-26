import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Layered Correlation Histogram
# Used for: Classification tasks with quantitative features.
# 
# This function takes a dataframe, a response variable (target), and an explanatory variable (predictor).
# It will then generate a histogram showing the distribution of the target variable grouped by the target variable,
# for the purposes of exploratory data analysis. Output graphs are stored in the visualizations/exploration/ folder
# of the project.
# 
# df: Dataframe that will be processed.
# 
# target: Response variable column.
# 
# predictor: Explanatory variable column.
# 
# predictor/target_mask: Optional names for predictor and response in chart labels. Default is to match
#   predictor and target column names exactly.
# 
# style: PLT style to use. Defaults to default style.
# 
# dpi: DPI of image to save. Default 160.
# 
# Fontsize: Font of all text in the graph. Default 10.

def corrHist(df,predictor,target,predictor_mask='',target_mask='',style='default',dpi=160,fontsize=10):
    if not predictor_mask:
        predictor_mask = predictor
    if not target_mask:
        target_mask = target
    plt.style.use(style)
    df = df.dropna(subset=[predictor])
    plt.figure(figsize=(10,8))
    for i in df[target].unique():
        plt.hist(df[df[target]==i][predictor],10,alpha=0.5,label=i)
    plt.legend(loc='upper right')
    plt.xlabel(predictor_mask)
    plt.ylabel('Frequency')
    plt.title('Histogram of {} grouped by {}'.format(predictor_mask,target_mask))
    plt.xticks(rotation=90)
    plt.savefig('visualizations/exploration/hist{}by{}.png'.format(predictor,target),dpi=dpi,bbox_inches='tight')

# Stacked Percentage Bar Chart
# Used for: Classification tasks with quantitative and qualitative features.
# 
# This function takes a dataframe, a response variable, and an explanatory variable.
# It will then generate a normalized stacked bar plot showing the percentage distribution of the target
# variable within the predictor variable for the purposes of exploratory data analysis.
# If the predictor is quantitative, this function will automatically separate it into ten bins before
# proceeding.
# 
# Arguments:
# 
# df: Dataframe that will be processed.
# 
# target: Response variable column.
# 
# predictor: Explanatory variable column.
# 
# predictor/target_mask: Optional names for predictor and response in chart labels. Default is to match
#   predictor and target column names exactly.
# 
# predictor/target_values_mask: Accepts a dictionary in the format:
# 
#     {predictor/target: {value1: new_name1, value2: new_name2, ...}}
# 
#   For example, in the Titanic dataset:
#   
#     {'Survived': {0: 'Perished', 1: 'Survived'}}
# 
# style: PLT style to use. Defaults to default style.
# 
# verbose: Adds number labels to bar slices. Default true.
# 
# dpi: DPI of image to save. Default 160.
# 
# Fontsize: Font of all text in the graph. Default 10.

def percBar(df,predictor,target,predictor_mask='',target_mask='',predictor_values_mask='',target_values_mask='',style='default',verbose=True,dpi=160,fontsize=10):
    df = df.copy()
    if(df[predictor].dtype!=object):
        df[predictor] = pd.cut(df[predictor],10,labels=False)
    if not predictor_mask:
        predictor_mask = predictor
    if not target_mask:
        target_mask = target
    if predictor_values_mask:
        df = df.replace(predictor_values_mask)
    if target_values_mask:
        df = df.replace(target_values_mask)
    plt.style.use(style)
    plt.rc('font',size=fontsize)
    df = df.dropna(subset=[predictor])
    lengths = []
    labels = np.append(df[predictor].unique(),target)
    if(len(labels) > 25):
        print('Error: Target variable ' + str(target) + ' contains more than 25 variants. Terminating procedure.')
        return
    plt.figure(figsize=(1.25*len(labels),6))
    for j in df[target].unique():
        for i in df[predictor].unique():
            # For each variant in the target variable:
            # For each variant in the predictor variable, save the number of times that the target variant and predictor variant appear together.
            lengths.append(df.loc[df[predictor]==i].loc[df[target]==j][target].count()/df.loc[df[predictor]==i][target].count())
        # At the end, append the number of the target variant divided by the total number of target instances.
        # This gives us the native distribution of the target variable, which allows us to compare how different the various
        # predictor categories are.
        lengths.append(df.loc[df[target]==j][target].count()/df[target].count())
    n = df[predictor].value_counts().values # Store the count of each prediction category, in case the user wants them displayed.
    n = np.append(n,df[target].count())
    for k in range(0,len(df[target].unique())):
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
        plt.bar(np.where(labels==target,target_mask,labels),lengths[k*len(labels):(k+1)*len(labels)],bottom=previousData,label=df[target].unique()[k])
        if verbose:
            labels = np.where(labels==target,target_mask,labels)
            for m in range(0,len(labels)):
                plt.text(labels[m],np.add(lengths[k*len(labels):(k+1)*len(labels)],previousData)[m],round(100*lengths[k*len(labels):(k+1)*len(labels)][m],2),ha="center",va="top")
                if(k==len(df[target].unique())-1):
                    plt.text(labels[m],np.add(lengths[k*len(labels):(k+1)*len(labels)],previousData)[m],"N=" + str(n[m]),ha="center",va="bottom")
    plt.legend(bbox_to_anchor=(1.01,1.01),loc='upper left')
    plt.xlabel(predictor_mask)
    plt.ylabel('Distribution')
    if verbose:
        plt.title('Distribution of {} within {}'.format(target_mask,predictor_mask),pad=15.0)
    else:
        plt.title('Distribution of {} within {}'.format(target_mask,predictor_mask))
    plt.xticks(rotation=90)
    plt.savefig('visualizations/exploration/perc{}by{}.png'.format(predictor,target),dpi=dpi,bbox_inches='tight')