# 🚗 Vehicle Emissions Analysis: A Data Science Journey

<p align="center">
  <img src="https://i.postimg.cc/NG542KhM/Vehicle-Emissions-Banner.png" alt="Vehicle Emissions Banner" width="800"/>
</p>

<div align="center">
  
  ![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
  ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-orange.svg)
  ![License](https://img.shields.io/badge/License-MIT-green.svg)
  
</div>

## 📊 Project Highlights

This project compares vehicle characteristics and their relationship to CO2 emissions between 2014 and 2025 data, revealing significant technological evolution in the automotive industry.

### Key Findings:

| Model Type | 2014 R² | 2025 R² | Change |
|------------|---------|---------|--------|
| Engine Size | 0.68 | 0.31 | -0.37 |
| Fuel Consumption | 0.76 | 0.95 | +0.19 |
| Cylinders | 0.57 | 0.21 | -0.36 |
| Multiple Regression | 0.84 | 0.95 | +0.11 |

<p align="center">
  <img src="https://i.postimg.cc/NMDsZS30/model-comparison.png" alt="Model Comparison" width="600"/>
</p>

## 🔍 The Discovery

What started as a simple machine learning practice project led to an unexpected discovery: the relationship between vehicle characteristics and CO2 emissions has fundamentally changed between 2014 and 2025.

In 2014, engine size was a strong predictor of emissions (R²=0.68). By 2025, this relationship had weakened significantly (R²=0.31), while fuel consumption became an extremely reliable predictor (R²=0.95).

<p align="center">
  <img src="https://i.postimg.cc/1Xd93PCW/overlap-comparison-2x2.png" alt="Scatterplot Comparison" width="700"/>
</p>

## 🧠 The Science Behind It

### Linear Regression Models

This project uses simple and multiple linear regression models to establish relationships between vehicle characteristics and CO2 emissions:

$$ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \epsilon $$

Where:
- $y$ represents CO2 emissions
- $x_i$ represents vehicle characteristics (engine size, cylinders, fuel consumption)
- $\beta_i$ represents coefficients determined through training
- $\epsilon$ represents error term

### Model Evaluation

Models are evaluated using the coefficient of determination (R²), which measures the proportion of variance in the dependent variable explained by the independent variables:

$$ R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2} $$

Where:
- $y_i$ is the actual value
- $\hat{y}_i$ is the predicted value
- $\bar{y}$ is the mean of actual values

## 🚀 Technological Evolution Explained

The dramatic shift in predictive relationships between 2014 and 2025 reveals the automotive industry's technological transformation:

<p align="center">
  <img src="https://i.postimg.cc/3NkT6V82/Technological-Evolution-Banner.png" alt="Technological Evolution" width="600"/>
</p>

1. **Engine Decoupling**: Advanced technologies like turbocharging, direct injection, and variable valve timing have decoupled engine size from emissions performance.

2. **Standardized Efficiency**: The strengthened relationship between fuel consumption and emissions indicates standardized conversion efficiency across manufacturers.

3. **Regulatory Impact**: Emissions regulations have pushed manufacturers to develop technologies that maximize efficiency regardless of engine size.

4. **Measurement Methods**: The near-perfect correlation in 2025 may also reflect changes in how emissions are measured or calculated.

## 📈 Visualizing the Relationships

### 2014 Models

<p align="center">
  <img src="https://i.postimg.cc/ZnNTn2G4/Figure-2014.png" alt="2014 Regression Models" width="700"/>
</p>

### 2025 Models

<p align="center">
  <img src="https://i.postimg.cc/3RBqBB2m/Figure-2025.png" alt="2025 Regression Models" width="700"/>
</p>

## 💡 Implications

1. **For Consumers**: Engine size has become a less reliable indicator of a vehicle's environmental impact. Fuel consumption ratings should be the primary consideration.

2. **For Policymakers**: Fuel economy standards have successfully created a direct link between fuel consumption and emissions, regardless of engine design.

3. **For Manufacturers**: The industry has successfully employed technological solutions that decouple engine physical characteristics from emissions performance.

4. **For Data Scientists**: This case demonstrates how data analysis can reveal underlying technological trends and their impacts on established relationships.

## 🛠️ Technologies Used

- **Python**: Core programming language
- **scikit-learn**: For implementing regression models
- **pandas**: For data manipulation and analysis
- **matplotlib & seaborn**: For data visualization
- **numpy**: For numerical computing

## 🔧 Running the Analysis

```bash
# Clone the repository
git clone https://github.com/your-username/vehicle-emissions-analysis.git
cd vehicle-emissions-analysis

# Install dependencies
pip install -r requirements.txt

# Run the analysis for both datasets
python src/vehicle_emissions_2014.py
python src/vehicle_emissions_2025.py
```

## 📝 My Learning Journey

This project represents a significant milestone in my data science learning journey. What began as a practice exercise in linear regression became an exciting discovery about the evolution of automotive technology.

Key learning points:
- Practical application of linear regression models
- Interpretation of R² scores and their significance
- Data visualization techniques for regression analysis
- Connecting statistical findings to real-world technological changes

## 📚 References

1. IBM Developer Skills Network. (2014). *Fuel Consumption and CO2 Emissions Dataset*.
2. Scikit-learn documentation: [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
3. U.S. Environmental Protection Agency. (2023). *Automotive Trends Report*.
4. International Council on Clean Transportation. (2024). *Global Fuel Economy Initiative Status Report*.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.