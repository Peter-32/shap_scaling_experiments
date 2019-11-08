class RunEverything:
    # Common imports
    import pandas as pd
    import numpy as np
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    import shap

    # Import from my GitHub
    from getxgboostmodel.getxgboostmodel import get_xgboost_model
    from randomizedgridsearch.randomizedgridsearch import RandomizedGridSearch
    from transformers.transformers import *

    import sys

    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")


    # Example dataset
    boston_data = load_boston()

    # Extract pandas dataframe and target
    X = pd.DataFrame(boston_data['data']).copy().values
    y = pd.DataFrame(boston_data['target']).copy().values

    # Train/test split
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.20, random_state=42)
    train_X, test_X = train_X, test_X
    train_y, test_y = train_y.reshape(-1, 1), test_y.reshape(-1, 1)

    # An okay model fit to the data
    try:
        xgb_model
    except:
        xgb_model = get_xgboost_model(train_X, train_y)

    # Pipeline
    pipe = Pipeline([('standard_scaler', StandardScalerTransform()),
                     ('min_max_scaler', MinMaxScalerTransform()),
                     ('binarizer', BinarizerTransform()), 
                     ('k_bins_discretizer', KBinsDiscretizerTransform()),
                     ('k_bins_discretizer2', KBinsDiscretizerTransform2()),
                     ('k_bins_discretizer3', KBinsDiscretizerTransform3()),
                     ('kernel_centerer', KernelCentererTransform()),
                     ('model', xgb_model)])

    # Find the number of features
    num_features = train_X.shape[1]

    # Testing with these indices
    indices = list(range(num_features))

    # Scalers
    scaler_values = [0.05]*num_features

    # Possible configurations [None, True, or False] - None means not decided yet
    param_distributions = {
        'standard_scaler': scaler_values,
        'min_max_scaler': scaler_values,
        'binarizer': scaler_values,
        'k_bins_discretizer': scaler_values,
        'k_bins_discretizer2': scaler_values,
        'k_bins_discretizer3': scaler_values,
        'kernel_centerer': scaler_values,
    }

    experiments_results = pd.DataFrame()    



    for iteration in range(1):
        # Randomly search the space n_iter times
        experiments_results_temp = RandomizedGridSearch(
            n_experiments=100,
            pipe=pipe,
            param_distributions=param_distributions,
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
            test_y=test_y,
            scoring='neg_mean_squared_error')

        # Append to experiment results
        experiments_results = experiments_results.append(experiments_results_temp, ignore_index=True)

        # Ignore past data
        # experiments_results = experiments_results_temp

        # Drop score
        experiments_X_df = experiments_results.drop(['score'], axis=1)

        # Get column names
        X_column_names = experiments_X_df.columns

        # Convert to numpy
        experiments_X = experiments_X_df.values
        experiments_y = experiments_results[['score']].values

        # Create an XGBoost model tuned with the experiments data
        try:
            xgb_experiments_model
            # Tune hyperparameters every once in a while
            # if iteration % 7 == 6:
            #     xgb_experiments_model = get_xgboost_model(experiments_X, experiments_y)
        except:
            xgb_experiments_model = get_xgboost_model(experiments_X, experiments_y)

        # Fit the model
        xgb_experiments_model.fit(experiments_X_df, experiments_y)

        # Extract shap values
        explainer = shap.TreeExplainer(xgb_experiments_model)
        shap_values = explainer.shap_values(experiments_X_df)

        # Shap as dataframe
        shap_values_of_experiments = pd.DataFrame(shap_values, columns=X_column_names)
        shap_values_of_experiments['score'] = experiments_y

        # Function to support analysis
        def find_significance_from_experiments_results(importance_threshold=0.001, max_toggles_to_lock_per_series=5):
            temp_df = shap_values_of_experiments.drop(['score'], axis=1).copy()
            for i in range(0, len(shap_values_of_experiments.index)):
                for j in range(0, len(shap_values_of_experiments.columns)):
                    if not experiments_results.iloc[i, j]:
                        temp_df.iloc[i, j] = -1 * shap_values_of_experiments.iloc[i, j]
            temp_df = temp_df.sum().sort_values()
            options_to_set_to_false = temp_df[temp_df > 0]
            options_to_set_to_true = temp_df[temp_df < 0]    
            sum_value = (options_to_set_to_false.sum() + abs(options_to_set_to_true.sum()))
            options_to_set_to_false = options_to_set_to_false / sum_value
            options_to_set_to_true = abs(options_to_set_to_true) / sum_value
            options_to_set_to_false = options_to_set_to_false[options_to_set_to_false > importance_threshold].sort_values(ascending=False)
            options_to_set_to_true = options_to_set_to_true[options_to_set_to_true > importance_threshold]
            return options_to_set_to_false[0:max_toggles_to_lock_per_series], options_to_set_to_true[0:max_toggles_to_lock_per_series]

        # Call function
        options_to_set_to_false, options_to_set_to_true = find_significance_from_experiments_results()

        # Make the set to true DF
        options_to_set_to_true_df = pd.DataFrame()
        transformation, value = None, None
        try:
            transformation_and_value = options_to_set_to_true.keys()
        except:
            transformation, value = [], []
        if len(transformation_and_value) > 0:
            options_to_set_to_true_df["transformation"] = [x.split("__")[0] for x in transformation_and_value]
            options_to_set_to_true_df["value"] = [x.split("__")[1] for x in transformation_and_value]
        else:
            options_to_set_to_true_df["transformation"] = []
            options_to_set_to_true_df["value"] = []
        options_to_set_to_true_df["significance"] = options_to_set_to_true.values

        # Make the false DF
        options_to_set_to_false_df = pd.DataFrame()
        transformation, value = None, None
        try:
            transformation_and_value = options_to_set_to_false.keys()
        except:
            transformation, value = [], []
        if len(transformation_and_value) > 0:
            options_to_set_to_false_df["transformation"] = [x.split("__")[0] for x in transformation_and_value]
            options_to_set_to_false_df["value"] = [x.split("__")[1] for x in transformation_and_value]
        else:
            options_to_set_to_false_df["transformation"] = []
            options_to_set_to_false_df["value"] = []
        options_to_set_to_false_df["significance"] = options_to_set_to_false.values

        # Reset updates
        updates = 0

        # Set to True
        for index, row in options_to_set_to_true_df.iterrows():            
            if param_distributions[row['transformation']][int(row['value'])] < 1.0:
                updates += 1
            param_distributions[row['transformation']][int(row['value'])] *= 2.0        
            if param_distributions[row['transformation']][int(row['value'])] >= 1.0:
                param_distributions[row['transformation']][int(row['value'])] = 1.0        


        for index, row in options_to_set_to_false_df.iterrows():
            if param_distributions[row['transformation']][int(row['value'])] > 0.01:
                updates += 1        
            param_distributions[row['transformation']][int(row['value'])] *= 0.5
            if param_distributions[row['transformation']][int(row['value'])] <= 0.01:
                param_distributions[row['transformation']][int(row['value'])] = 0.01

        # End early if there is nothing to tune
        print("Updates:", updates)
        print("best score: ", min(experiments_results['score']), "\nparams: ", param_distributions)    
        if updates == 0:
            break
    print("Done")