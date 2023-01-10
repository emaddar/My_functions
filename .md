```py

def get_index_to_remove_by_Cooks_Distance(X_train, y_train, preprocessor):
    # Fit the transformer to the training data
    preprocessor.fit(X_train)
    X_test_pipe = preprocessor.transform(X_train)


    # Get the names of the columns added by the OneHotEncoder
    new_columns = preprocessor.get_feature_names_out()
    new_columns = [w.replace('pipeline-1__', '') for w in new_columns]
    new_columns = [w.replace('pipeline-2__', '') for w in new_columns]


    newdf = pd.DataFrame(X_test_pipe)

    newdf.columns = new_columns


    X = sm.add_constant(newdf)
    X = X.set_index(y_train.index)
    estimation = sm.OLS(y_train, X_test_pipe).fit()

    influence = estimation.get_influence().cooks_distance[0]

    X['dcooks'] = influence
    n = X.shape[0]
    p = X.shape[1]
    seuil_dcook = 4/(n-p)

    index_to_be_removed = X[X['dcooks']>seuil_dcook].index

    #plt.figure(figsize=(10,6))
    #plt.bar(X.index, X['dcooks'])
    #plt.xticks(np.arange(0, len(X), step=int(len(X)/10)))
    #plt.xlabel('Observation')
    #plt.ylabel('Cooks Distance')
    # Plot the line
    #plt.hlines(seuil_dcook, xmin=0, xmax=len(X_train), color='r')
    #plt.show()
    
    return index_to_be_removed
```
