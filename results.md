# Key Findings

## Model Performance (R² values)

| Model Type | 2014 Data | 2025 Data |
|------------|-----------|-----------|
| Engine Size | 0.68 | 0.31 |
| Fuel Consumption | 0.76 | 0.95 |
| Cylinders | 0.57 | 0.21 |
| Multiple Regression | 0.84 | 0.95 |

## Observations
   
1. In 2014 data, all features contributed meaningfully to predicting emissions
2. In 2025 data, fuel consumption became the dominant predictor (R²=0.95)
3. Engine size and cylinders became much weaker predictors in 2025
   
These findings suggest significant technological evolution in vehicle design between 2014 and 2025.

## Implications

1. **For Consumers**: Engine size has become a less reliable indicator of environmental impact by 2025. Fuel consumption ratings are now the dominant factor to consider.

2. **For Policymakers**: Fuel economy standards have been effective in creating a direct link between fuel consumption and emissions, regardless of engine design.

3. **For Manufacturers**: The industry has increasingly employed technological solutions that decouple engine physical characteristics from emissions performance.

4. **For Data Analysis**: The extremely high correlation between fuel consumption and CO2 emissions (0.95) raises questions about whether emissions are directly measured or calculated from fuel consumption using a standard formula.

## Visual Evidence

The regression plots clearly show:
1. Much tighter clustering along the regression line for fuel consumption in 2025
2. More scattered relationship for engine size and cylinders in 2025 compared to 2014
3. Multiple regression model performance in 2025 essentially matches the fuel consumption model alone
