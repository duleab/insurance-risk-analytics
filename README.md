
## Setup Instructions
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Initialize DVC: `dvc init`
4. Run DVC pipeline: `dvc repro`
5. Follow the workflow in notebooks directory

## Dataset Overview
- **Size**: 10,000 insurance policies
- **Features**: 11 variables including demographics, vehicle details, and financial metrics
- **Date Range**: February 2014 to June 2041
- **Data Quality**: Complete dataset with no missing values
- **Memory Usage**: 2.63 MB

## Key Findings

### Risk Metrics Summary
- **Overall Loss Ratio**: 9.2% (Excellent profitability)
- **Claim Frequency**: 14.5% (Reasonable occurrence rate)
- **Average Claim Amount**: $3,023 (Moderate severity)
- **Average Premium**: $5,162 (Good premium-to-claim ratio)

### Geographic Distribution
- **Western Cape**: 30.5% (3,053 policies) - Highest concentration
- **Gauteng**: 25.5% (2,553 policies) - Major economic center
- **KwaZulu-Natal**: 19.8% (1,981 policies)
- **Eastern Cape**: 14.5% (1,452 policies)
- **Free State**: 9.6% (961 policies)

### Vehicle Portfolio
- **Sedan**: 40.3% (4,032 policies) - Dominant vehicle type
- **SUV**: 25.2% (2,524 policies)
- **Hatchback**: 20.1% (2,007 policies)
- **Truck**: 9.4% (938 policies)
- **Sports**: 5.0% (499 policies)

### Statistical Testing Results
- **Province Risk Differences**: No statistically significant differences (p > 0.05)
- **Gender Risk Differences**: No statistically significant differences (p > 0.05)
- **Vehicle Make Risk Differences**: No statistically significant differences (p > 0.05)
- **Vehicle Type Risk Differences**: No statistically significant differences (p > 0.05)
- **Postal Region Risk Differences**: No statistically significant differences (p > 0.05)

## Analysis Components

### 1. Exploratory Data Analysis ✅ COMPLETED
- Portfolio performance metrics
- Risk factor analysis
- Temporal and geographical patterns
- Distribution analysis of all variables
- Risk metrics calculation

### 2. Statistical Testing ✅ COMPLETED
- ANOVA tests for continuous variables
- Chi-square tests for categorical variables
- Hypothesis testing for risk variations across:
  - Provinces
  - Postal regions
  - Gender
  - Vehicle make
  - Vehicle type
- Effect size analysis

### 3. DVC Pipeline ✅ COMPLETED
- Automated data processing pipeline
- Feature engineering automation
- Statistical testing automation
- Model training pipeline
- Results generation and recommendations
- Pipeline health score: 100%

### 4. Predictive Modeling ✅ COMPLETED
- Claims prediction models
- Premium optimization framework
- Risk scoring algorithms

## Business Recommendations

Based on statistical analysis:

1. **Current Pricing Model**: Well-calibrated across all tested factors
2. **Geographic Strategy**: No significant provincial risk differences support uniform geographic pricing
3. **Demographic Approach**: Gender-neutral pricing is statistically justified
4. **Vehicle Segmentation**: Current vehicle-based pricing appears appropriate
5. **Portfolio Optimization**: Focus on the 85.5% zero-claim segment for retention strategies
6. **Profitability**: Excellent 9.2% loss ratio indicates strong underwriting performance

## Technology Stack
- **Python**: Data analysis and modeling
- **Pandas & NumPy**: Data manipulation
- **Matplotlib & Seaborn**: Data visualization
- **Plotly**: Interactive visualizations
- **SciPy & Statsmodels**: Statistical testing
- **DVC**: Data version control and pipeline management
- **Git**: Version control
- **Jupyter**: Interactive development

