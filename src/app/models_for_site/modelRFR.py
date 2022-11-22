from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import pickle
try:
    from .model1 import get_data_for_regression
except:
    from model1 import get_data_for_regression

from sklearn.model_selection import RandomizedSearchCV

def get_grid():
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 20, stop = 200, num = 20)]

    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(3, 33, num = 3)]
    max_depth.append(None)

    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'bootstrap': bootstrap}
    return random_grid

def main_processing_RFR(df: pd.DataFrame = None, name_y: str= None):
    if name_y is None:
        name_y = "Прочность при растяжении, МПа"  #"Модуль упругости при растяжении, ГПа"
    if df is None:
        df = pd.read_csv(r"data/dataForFinal.csv")

    x_train, x_test, y_train, y_test = get_data_for_regression(df, name_y)
    model = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    random_grid = get_grid()
    rf_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(x_train, y_train)
    model = rf_random.best_estimator_
    filename = 'static/finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    print("best_params: ", rf_random.best_params_)
    print("best_score: ", rf_random.best_score_)
    pr = model.predict(x_test).ravel()
    rel = np.abs(pr - y_test)/y_test
    print(f"Test relative error, mean: {np.mean(rel)}, max: {np.max(rel)}")
    return {"rel": rel.tolist(), "pr": pr.tolist(), "y_test": y_test.tolist(), "model_path": filename, "N": pr.shape[0]}

if __name__ == "__main__":
    main_processing_RFR()