#  Telco Customer Churn: End-to-End Machine Learning & BI Pipeline

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Tableau](https://img.shields.io/badge/Tableau-E97627?style=for-the-badge&logo=Tableau&logoColor=white)

##  Project Overview
Customer retention is a critical driver of profitability in the telecommunications sector. This project bridges the gap between predictive data science and actionable business intelligence. 

I built an end-to-end machine learning pipeline that predicts customer churn probability and translates those predictions into financial "Expected Value" loss. The final output is an interactive dashboard that allows retention managers to prioritize high-value, high-risk customers.

**👉 [View the Interactive Business Dashboard on Tableau Public](https://public.tableau.com/views/customerchurndashboard_17736076043470/Dashboard1?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)**

##  Methodology & Pipeline

### Phase 1: Exploratory Data Analysis (EDA)
* Analyzed the Telco Customer Churn dataset using **Pandas** and **Seaborn** to identify historical drivers of churn.
* Identified key risk segments (e.g., Month-to-Month contracts and Fiber Optic services).

### Phase 2: Data Preprocessing & Feature Engineering
* **Encoding:** Converted categorical variables (gender, contract type) into machine-readable numeric formats (One-Hot Encoding).
* **Scaling:** Normalized continuous financial variables (Monthly Charges, Total Charges) to ensure model stability.
* **Splitting:** Segmented data into an 80/20 train-test split to evaluate unseen data accurately.

### Phase 3: Machine Learning Model (The Engine)
Trained a **Random Forest Classifier** to predict the likelihood of churn.
* **Optimization Strategy:** In a business context, the cost of a "False Negative" (missing a churning customer) far exceeds a "False Positive" (offering a discount to a safe customer). Therefore, the model was optimized for **Recall**.
* **Probability Scoring:** Instead of binary (1/0) outputs, the model extracts the exact `Churn Probability %` for each user to enable dynamic risk thresholds.

### Phase 4: Business Intelligence (The Product)
Exported the ML predictions to **Tableau** to calculate **Revenue at Risk** (Monthly Charges × Churn Probability). 
The final dashboard features:
1. **Revenue at Risk Treemap:** Visualizing financial exposure by contract type.
2. **Segment Averages Dual-Axis Chart:** Contrasting total financial risk against average churn probabilities.
3. **Dynamic Call List:** A customer-level priority table controlled by a custom AI Risk Threshold parameter.

##  Repository Structure
* `scripts/` -  Contains the python file (.py)  for EDA, Preprocessing, and Random Forest modeling.
* `data/` - Raw Telco dataset and the exported CSV containing the AI probability predictions.
* `README.md` - Project documentation.

##  How to Run
1. Clone the repository: git clone [https://github.com/YourUsername/telco-customer-churn-prediction]
2. Install the required dependencies: 
   ```bash
   pip install -r requirements.txt
3. Run the script to generate predictions and export the final dataset:
   python scripts/churn_model.py
