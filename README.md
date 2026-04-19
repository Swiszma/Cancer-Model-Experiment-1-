# Cancer-Model-Experiment-1
## A Machine Learning Model to diagnose breast tumors as either Malignant (1) or Benign (0).


### Table of Content
- [Project Overview](#project-overview)
- [Data Auditing and Preprocessing](#data-auditing-and-preprocessing)
- [Escaping Data Leakage and Scaling Traps](#escaping-data-leakage-and-scaling-traps)
- [Baseline Evaluation and The Ethical Dilemma](#baseline-evaluation-and-the-ethical-dilemma)
- [Threshold Optimization](#threshold-optimization)
- [Model Evaluation](#model-evaluation)
- [Project Improvement using Hyperparameter Tuning](#project-improvement-using-hyperparameter-tuning)
- [Explainable AI and Feature Reduction](#explainable-ai-and-feature-reduction)
- [Final Evaluation and Deployment](#final-evaluation-and-deployment)
- [Resources](#resources)

### Project Overview
The objective of this project was to build a Machine Learning classification model to diagnose breast tumors as either Malignant (1) or Benign (0) using the renowned Wisconsin Breast Cancer Dataset. Rather than simply deploying a baseline algorithm, this project focused heavily on rigorous data hygiene, preventing data leakage, and applying medical-domain ethics to optimize the model for a 100% Recall rate (zero missed cancers).

### Data Auditing and Preprocessing
Before any modeling occurred, the dataset (569 patients, 30+ cell features) underwent strict Exploratory Data Analysis (EDA) and cleaning:
- Sanity Checks: I deployed a Seaborn missing-data heatmap and a duplicate-row scanner, confirming the dataset was pristine (0 NaNs, 0 duplicates).
Feature Selection: I explicitly dropped the id column, recognizing that patient ID numbers possess zero mathematical correlation to tumor malignancy and would only introduce noise.
- Target Mapping: I mapped the categorical text (M and B) to binary integers (1 and 0), consciously assigning the Malignant class to 1 (Positive) because detecting the deadly disease was the primary objective.

### Escaping Data Leakage and Scaling Traps
I successfully navigated two major pitfalls that commonly ruin classification models:
- The Scaling Trap: I identified that features like area_mean (values > 1000) would completely overpower features like smoothness_mean (values < 0.1) during Gradient Descent. I applied StandardScaler to ensure the Cost Function "bowl" was perfectly symmetrical.
- Preventing Data Leakage: To maintain mathematical purity, I strictly applied train_test_split (with stratify=y to maintain the 63/37 class balance) first. I then fit the scaler exclusively to the Training data, ensuring the Test vault remained 100% unseen and uncompromised.

### Baseline Evaluation and The Ethical Dilemma
The baseline Logistic Regression model achieved an outstanding 97.08% Accuracy.
However, a deeper inspection of the Confusion Matrix revealed a critical flaw: the model produced 4 False Negatives. In an oncology setting, sending 4 cancer patients home with a clean bill of health is a catastrophic liability. However, in this specific domain, Accuracy and Precision are secondary to Recall (because the focus is "how manay of the cancer patients is the model able to perfectly detect").

### Threshold Optimization
To eliminate the False Negatives, I bypassed Scikit-Learn's default 50% probability threshold to built a custom Threshold Analyzer which iterates through extreme thresholds (20%, 15%, 10%, 5%, and 1%).
- The Business Decision: I discovered that dropping the decision threshold to 5% successfully pushed the Recall to 100% (0 missed cancers).
- The Trade-off: The mathematical cost was an increase in False Positives (predicting Malignant for healthy patients) to roughly 13%. I concluded this was a highly acceptable trade-off, as a 13% false-alarm/biopsy rate perfectly aligns with real-world human radiology standards. Saving lives justified the drop in Precision

### Model Evaluation
To mathematically validate the threshold decision, I mapped the model's performance across all possible thresholds using an ROC Curve.
The model achieved an exceptional AUC (Area Under the Curve) Score of 99.75%, proving that it possesses near-perfect intelligence in separating benign cells from malignant ones.
I plotted our custom 5% threshold directly onto the ROC curve, visually demonstrating to stakeholders how we maximized the True Positive Rate while minimizing the False Positive Rate.


## Project Improvement using Hyperparameter Tuning

To eliminate False Negatives without causing massive false-alarm rates, I bypassed standard manual threshold adjustments by engineering a fundamentally smarter model:
- Medical Ethics in Math: I applied class_weight='balanced' to the Logistic Regression algorithm, heavily multiplying the error penalty for missing a cancer patient.
- GridSearchCV: I automated the testing of 60 different hyperparameter combinations, explicitly instructing the model (scoring='recall') that the only metric that mattered for winning the tournament was maximizing the True Positive Rate.

### Explainable AI and Feature Reduction
When extracting the coefficients from the tuned champion model, I uncovered a fascinating biological and mathematical reality:
- The Grid Search selected the L1 (Lasso) Penalty, which acts as a ruthless mathematical filter.
- The model identified massive multicollinearity (e.g., Area, Radius, and Perimeter all representing "Size") and noise. It mathematically deleted 28 out of 30 features (shrinking their slopes to exactly 0.0)...sounds crazy but essential. The deleted features says the same thing as the remaining 2 features, making the 28 deleted features unnecessary.
- The model achieved its results by looking at only two extreme biological markers: The maximum size of the tumor boundary (perimeter_worst) and the maximum severity of its jagged spikes (concave points_worst).

### Final Evaluation and Deployment
By fixing the Cost Function bowl and tuning the hyperparameters, the final model was a resounding success:
- The Results: At the standard 50% decision threshold, the model achieved a staggering 100% Recall (0 missed cancers) while generating only 5 False Positives.
- ROC/AUC: The model achieved an exceptional AUC Score of 99.75%, proving near-perfect intelligence in separating benign from malignant cells.
- Serialization: The final Tuned Model and the StandardScaler were successfully serialized into .pkl files using joblib, making the pipeline fully ready for software engineering integration and clinical deployment.

### Resources
See codes [here](https://drive.google.com/file/d/1bWq01hbDMN-wqfuHKzam8vUjIzzUkLbR/view?usp=drive_link)
