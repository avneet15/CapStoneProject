#extract all cuisines
#create a matrix with restaurat id as rows and cuisines as cols
import pandas as pd
import numpy as np
import os


pd.set_option('display.mpl_style', 'default') # Make the graphs a bit prettier


def get_cuisine_info():
    base_dir = os.path.dirname(os.path.realpath('__file__'))
    business_file_path = os.path.join(base_dir, "yelp_boston_academic_dataset/yelp_academic_dataset_business.json")

    with open(business_file_path, 'r') as business_file:
        # the file is not actually valid json since each line is an individual
        # dict -- we will add brackets on the very beginning and ending in order
        # to make this an array of dicts and join the array entries with commas
        business_json = '[' + ','.join(business_file.readlines()) + ']'

    df = pd.read_json(business_json)

    #df = pd.read_json("/Users/avneet/Desktop/MASTER's PROJECT/CS246 Submission/yelp_boston_academic_dataset/yelp_academic_dataset_business.json",orient='columns')
    df_categories_rest = df[['categories','business_id']]

    #print(df_categories_rest.head(2))

    #get unique categories
    df2=df['categories']
    df2 = df2.values.flatten().tolist() #form a list
    df2 = [item for sublist in df2 for item in sublist] #flatten the list
    df2 = list(set(df2)) #remove duplicatesa


    #form a matrix with unique cat and business id
    df = pd.DataFrame(np.random.rand(len(df_categories_rest),len(df2)), index=df_categories_rest['business_id'], columns=df2)


    #print(df.shape)
    #dataframe initialized with 0 values
    df[:] = 0

    #one hot encoding
    for i in range(len(df_categories_rest)):
        val = df_categories_rest['categories'][i]
        for v in val:
            df[v][i]=1

    df['business_id'] = df.index
    return df

get_cuisine_info()