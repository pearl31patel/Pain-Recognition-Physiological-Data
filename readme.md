# Pain Recognition Using Physiological Data

This project is a machine learning system for recognizing pain using physiological signals collected from wearable devices. The model uses features from diastolic blood pressure, systolic blood pressure, electrodermal activity (EDA), and respiration data to classify pain and no-pain conditions.

## Project Overview

The goal of this project is to classify whether a subject is experiencing pain based on physiological data. The dataset contains 60 subjects, and each subject has data for both pain and no-pain classes.

The project uses hand-crafted features such as:

- Mean
- Variance
- Minimum value
- Maximum value

These features are extracted from each physiological signal and used to train a machine learning classifier.

## Dataset

The dataset contains the following columns:

- Subject ID
- Data Type
- Class
- Data

The available data types are:

| Data Type | Description |
|---|---|
| `dia` | Diastolic Blood Pressure |
| `sys` | Systolic Blood Pressure |
| `eda` | Electrodermal Activity |
| `res` | Respiration |
| `all` | Fusion of all physiological signals |

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Random Forest Classifier

## Classifier Used

This project uses a Random Forest classifier. Random Forest was selected because it works well with small datasets, handles noise, reduces overfitting, and performs well with hand-crafted statistical features.

## Model Evaluation

The model is evaluated using 10-fold cross-validation. Each fold contains 6 subjects, and the same subject is not used in both training and testing.

The following metrics are calculated:

- Confusion Matrix
- Accuracy
- Precision
- Recall

## Results

The best performance was achieved using feature-level fusion with all physiological signals.

| Data Type | Accuracy | Precision | Recall |
|---|---:|---:|---:|
| All Signals | 0.733 | 0.731 | 0.750 |

Confusion Matrix:

```text
[[4.3, 1.7],
 [1.5, 4.5]]
```


## How to Run

```bash
python Project2.py <data_type> <data_file>
```

## Output

The script prints the following evaluation metrics:

- Confusion Matrix
- Classification Accuracy
- Precision
- Recall

The result files are saved inside the `results/` folder.

---

## Visualizations

This project includes visualizations for feature variability and physiological signal comparison.

### Feature Variability

Box plots are generated to show the variability of extracted features.

### Physiological Signal Plot

A line graph is generated to visually compare physiological signals such as blood pressure, EDA, and respiration.

---

## Conclusion

This project shows that physiological signals can be used to classify pain and no-pain conditions. The fusion of all signals gave the best performance because it combines information from blood pressure, EDA, and respiration. This makes the model more effective than using only one signal type.

---

## Author

**Pearl Viralkumar Patel**

![image alt](https://github.com/pearl31patel/Pain-Recognition-Physiological-Data/blob/3f4babefeae5d4dd0cf9c1350c5c450e35de5e57/pain_recognition.png)
