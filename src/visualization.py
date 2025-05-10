import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_regression_models(X_train, X_test, y_train, y_test, models, figsize=(15, 12)):
    """
    Plot regression models for different features.
    
    Parameters:
    -----------
    X_train, X_test : dict
        Dictionaries containing training and test features
    y_train, y_test : dict
        Dictionaries containing training and test target values
    models : dict
        Dictionary containing trained regression models
    figsize : tuple
        Figure size for the plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Engine Size plot
    ax1.scatter(X_train['engine'], y_train['engine'], color='blue', label='Training Data')
    ax1.scatter(X_test['engine'], y_test['engine'], color='green', label='Test Data')
    X_sorted = np.sort(np.concatenate((X_train['engine'], X_test['engine'])))
    y_pred_sorted = models['engine'].predict(X_sorted.reshape(-1, 1))
    ax1.plot(X_sorted, y_pred_sorted, '-r')
    ax1.set_xlabel("Engine Size")
    ax1.set_ylabel("CO2 Emission")
    ax1.set_title("Engine Size vs. CO2 Emission")
    ax1.legend()
    
    # Fuel Consumption plot
    ax2.scatter(X_train['fuel'], y_train['fuel'], color='blue', label='Training Data')
    ax2.scatter(X_test['fuel'], y_test['fuel'], color='green', label='Test Data')
    X_sorted = np.sort(np.concatenate((X_train['fuel'], X_test['fuel'])))
    y_pred_sorted = models['fuel'].predict(X_sorted.reshape(-1, 1))
    ax2.plot(X_sorted, y_pred_sorted, '-r')
    ax2.set_xlabel("Fuel Consumption")
    ax2.set_ylabel("CO2 Emission")
    ax2.set_title("Fuel Consumption vs. CO2 Emission")
    ax2.legend()
    
    # Cylinders plot
    ax3.scatter(X_train['cylinders'], y_train['cylinders'], color='blue', label='Training Data')
    ax3.scatter(X_test['cylinders'], y_test['cylinders'], color='green', label='Test Data')
    X_sorted = np.sort(np.concatenate((X_train['cylinders'], X_test['cylinders'])))
    y_pred_sorted = models['cylinders'].predict(X_sorted.reshape(-1, 1))
    ax3.plot(X_sorted, y_pred_sorted, '-r')
    ax3.set_xlabel("Cylinders")
    ax3.set_ylabel("CO2 Emission")
    ax3.set_title("Cylinders vs. CO2 Emission")
    ax3.legend()
    
    # Multiple Regression plot
    ax4.scatter(y_train['multiple'], models['multiple'].predict(X_train['multiple']), color='blue', label='Training Data')
    ax4.scatter(y_test['multiple'], models['multiple'].predict(X_test['multiple']), color='green', label='Test Data')
    min_val = min(min(y_train['multiple']), min(y_test['multiple']))
    max_val = max(max(y_train['multiple']), max(y_test['multiple']))
    ax4.plot([min_val, max_val], [min_val, max_val], 'r-', lw=2)
    ax4.set_xlabel("Actual CO2 Emission")
    ax4.set_ylabel("Predicted CO2 Emission")
    ax4.set_title("Multiple Regression: Actual vs Predicted")
    ax4.legend()
    
    fig.suptitle("Comparison of Regression Models", fontsize=16)
    plt.tight_layout()
    return fig

def plot_correlation_matrix(df):
    """Plot correlation matrix for the dataset"""
    correlation_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Variables')
    return plt.gcf()

def plot_residuals(y_true, y_pred, feature=None):
    """Plot residuals analysis"""
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals)
    ax1.axhline(y=0, color='r', linestyle='-')
    ax1.set_xlabel('Predicted CO2 Emissions')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted Values')
    
    # Residual distribution
    sns.histplot(residuals, kde=True, ax=ax2)
    ax2.set_xlabel('Residual Value')
    ax2.set_title('Distribution of Residuals')
    
    plt.tight_layout()
    return fig