def Negativedownsampling(train, ratio) :

    # Number of data points in the minority class
    number_records_fraud = len(train[train.isFraud == 1])
    fraud_indices = np.array(train[train.isFraud == 1].index)

    # Picking the indices of the normal classes
    normal_indices = train[train.isFraud == 0].index

    # Out of the indices we picked, randomly select "x" number (number_records_fraud)
    random_normal_indices = np.random.choice(normal_indices, number_records_fraud*ratio, replace = False)
    random_normal_indices = np.array(random_normal_indices)

    # Appending the 2 indices
    under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])

    # Under sample dataset
    under_sample_data = train.iloc[under_sample_indices,:]
    
    # Showing ratio
    print("Percentage of normal transactions: ", round(len(under_sample_data[under_sample_data.isFraud == 0])/len(under_sample_data),2)* 100,"%")
    print("Percentage of fraud transactions: ", round(len(under_sample_data[under_sample_data.isFraud == 1])/len(under_sample_data),2)* 100,"%")
    print("Total number of transactions in resampled data: ", len(under_sample_data))
    
    return under_sample_data

def reduce_mem_usage(df, verbose=True):
    import numpy as np
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df