import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

def train_basic_models(df, test_size=0.2, random_state=42):
    """
    Train basic regression models on the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset containing features and target
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing trained models, features, and performance metrics
    """
    results = {}
    
    # Extract features and target
    X = {}
    y = {}
    X_train = {}
    X_test = {}
    y_train = {}
    y_test = {}
    models = {}
    metrics = {}
    
    # Engine size model
    X['engine'] = df.ENGINESIZE.to_numpy()
    y['engine'] = df.CO2EMISSIONS.to_numpy()
    X_train['engine'], X_test['engine'], y_train['engine'], y_test['engine'] = train_test_split(
        X['engine'], y['engine'], test_size=test_size, random_state=random_state
    )
    
    models['engine'] = linear_model.LinearRegression()
    models['engine'].fit(X_train['engine'].reshape(-1, 1), y_train['engine'])
    
    y_pred = models['engine'].predict(X_test['engine'].reshape(-1, 1))
    metrics['engine'] = {
        'r2': r2_score(y_test['engine'], y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test['engine'], y_pred)),
        'mae': mean_absolute_error(y_test['engine'], y_pred)
    }
    
    # Fuel consumption model
    X['fuel'] = df.FUELCONSUMPTION_COMB.to_numpy()
    y['fuel'] = df.CO2EMISSIONS.to_numpy()
    X_train['fuel'], X_test['fuel'], y_train['fuel'], y_test['fuel'] = train_test_split(
        X['fuel'], y['fuel'], test_size=test_size, random_state=random_state
    )
    
    models['fuel'] = linear_model.LinearRegression()
    models['fuel'].fit(X_train['fuel'].reshape(-1, 1), y_train['fuel'])
    
    y_pred = models['fuel'].predict(X_test['fuel'].reshape(-1, 1))
    metrics['fuel'] = {
        'r2': r2_score(y_test['fuel'], y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test['fuel'], y_pred)),
        'mae': mean_absolute_error(y_test['fuel'], y_pred)
    }
    
    # Cylinders model
    X['cylinders'] = df.CYLINDERS.to_numpy()
    y['cylinders'] = df.CO2EMISSIONS.to_numpy()
    X_train['cylinders'], X_test['cylinders'], y_train['cylinders'], y_test['cylinders'] = train_test_split(
        X['cylinders'], y['cylinders'], test_size=test_size, random_state=random_state
    )
    
    models['cylinders'] = linear_model.LinearRegression()
    models['cylinders'].fit(X_train['cylinders'].reshape(-1, 1), y_train['cylinders'])
    
    y_pred = models['cylinders'].predict(X_test['cylinders'].reshape(-1, 1))
    metrics['cylinders'] = {
        'r2': r2_score(y_test['cylinders'], y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test['cylinders'], y_pred)),
        'mae': mean_absolute_error(y_test['cylinders'], y_pred)
    }
    
    # Multiple regression model
    X['multiple'] = np.column_stack((df.ENGINESIZE.to_numpy(), df.FUELCONSUMPTION_COMB.to_numpy()))
    y['multiple'] = df.CO2EMISSIONS.to_numpy()
    X_train['multiple'], X_test['multiple'], y_train['multiple'], y_test['multiple'] = train_test_split(
        X['multiple'], y['multiple'], test_size=test_size, random_state=random_state
    )
    
    models['multiple'] = linear_model.LinearRegression()
    models['multiple'].fit(X_train['multiple'], y_train['multiple'])
    
    y_pred = models['multiple'].predict(X_test['multiple'])
    metrics['multiple'] = {
        'r2': r2_score(y_test['multiple'], y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test['multiple'], y_pred)),
        'mae': mean_absolute_error(y_test['multiple'], y_pred)
    }
    
    return {
        'X': X,
        'y': y,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'models': models,
        'metrics': metrics
    }

def train_advanced_models(df, test_size=0.2, random_state=42):
    """
    Train advanced regression models on the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset containing features and target
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing trained models and performance metrics
    """
    # Extract features and target
    X = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']].values
    y = df['CO2EMISSIONS'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    results = {}
    
    # Polynomial regression
    poly_features = PolynomialFeatures(degree=2)
    X_poly_train = poly_features.fit_transform(X_train[:, 0].reshape(-1, 1))  # Only using engine size
    X_poly_test = poly_features.transform(X_test[:, 0].reshape(-1, 1))
    
    poly_model = linear_model.LinearRegression()
    poly_model.fit(X_poly_train, y_train)
    
    y_poly_pred = poly_model.predict(X_poly_test)
    
    results['polynomial'] = {
        'model': poly_model,
        'X_train': X_poly_train,
        'X_test': X_poly_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_poly_pred,
        'r2': r2_score(y_test, y_poly_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_poly_pred))
    }
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    rf_model.fit(X_train, y_train)
    
    y_rf_pred = rf_model.predict(X_test)
    
    results['random_forest'] = {
        'model': rf_model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_rf_pred,
        'r2': r2_score(y_test, y_rf_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_rf_pred)),
        'feature_importance': {
            'ENGINESIZE': rf_model.feature_importances_[0],
            'CYLINDERS': rf_model.feature_importances_[1],
            'FUELCONSUMPTION_COMB': rf_model.feature_importances_[2]
        }
    }
    
    return results