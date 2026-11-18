# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Import Required Libraries

2. Load and View the Dataset

3. Drop Unnecessary Columns

4. Convert Categorical Columns to category Type

5. Convert Categories to Numeric Codes

6. Separate Features (X) and Target (y)

7. Initialize Model Parameters (Weights)

8. Define the Sigmoid Function

9. Define the Loss (Cost) Function

10. Implement Gradient Descent

11. Define Prediction Function

12. Make Predictions and Compute Accuracy

13. Predict for New Students
```

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SANTHOSH S
RegisterNumber: 212224100052
```
```python
# -----------------------------------------
# EXPERIMENT 6: Logistic Regression using Gradient Descent (From Scratch)
# -----------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the Dataset
data = pd.read_csv('Placement_Data.csv')

print("Original Data:")
print(data.head())

# 2. Drop Unnecessary Columns
data = data.drop('sl_no', axis=1)
data = data.drop('salary', axis=1)

print("\nAfter dropping 'sl_no' and 'salary':")
print(data.head())

# 3. Convert Categorical Columns to 'category' Type
data["gender"] = data["gender"].astype('category')
data["ssc_b"] = data["ssc_b"].astype('category')
data["hsc_b"] = data["hsc_b"].astype('category')
data["degree_t"] = data["degree_t"].astype('category')
data["workex"] = data["workex"].astype('category')
data["specialisation"] = data["specialisation"].astype('category')
data["status"] = data["status"].astype('category')
data["hsc_s"] = data["hsc_s"].astype('category')

print("\nData types after converting to 'category':")
print(data.dtypes)

# 4. Convert Categories to Numeric Codes
data["gender"] = data["gender"].cat.codes
data["ssc_b"] = data["ssc_b"].cat.codes
data["hsc_b"] = data["hsc_b"].cat.codes
data["degree_t"] = data["degree_t"].cat.codes
data["workex"] = data["workex"].cat.codes
data["specialisation"] = data["specialisation"].cat.codes
data["status"] = data["status"].cat.codes
data["hsc_s"] = data["hsc_s"].cat.codes

print("\nData after converting categories to numeric codes:")
print(data.head())

# 5. Separate Features (X) and Target (y)
x = data.iloc[:, :-1].values   # all columns except last
y = data.iloc[:, -1].values    # last column (status)

print("\nFeature matrix X shape:", x.shape)
print("Target vector y shape:", y.shape)

# 6. Initialize Parameters
theta = np.random.randn(x.shape[1])  # Random initial weights for each feature

print("\nInitial theta (weights):")
print(theta)

# 7. Define Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 8. Define Loss Function (optional to print/check)
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h + 1e-10) + (1 - y) * np.log(1 - h + 1e-10))
    # Added small 1e-10 to avoid log(0)

# 9. Implement Gradient Descent
def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient

        # Optional: print loss every 100 iterations
        if (i + 1) % 100 == 0:
            current_loss = loss(theta, X, y)
            print(f"Iteration {i+1}, Loss: {current_loss:.4f}")

    return theta

# 10. Train the Model
theta = gradient_descent(theta, x, y, alpha=0.01, num_iterations=1000)

print("\nFinal theta (weights) after training:")
print(theta)

# 11. Define Prediction Function
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred

# 12. Make Predictions and Compute Accuracy
y_pred = predict(theta, x)
accuracy = np.mean(y_pred.flatten() == y)

print("\nTraining Accuracy:", accuracy)
print("Predicted labels (first 20):")
print(y_pred[:20])

# 13. Predict for New Students
# NOTE: The order of features must match 'x' columns exactly.
# Example new students (values must match the encoded format used above)
xnew1 = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
xnew2 = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])

y_prednew1 = predict(theta, xnew1)
y_prednew2 = predict(theta, xnew2)

print("\nPrediction for new student 1 (0 = Not Placed, 1 = Placed):", y_prednew1[0])
print("Prediction for new student 2 (0 = Not Placed, 1 = Placed):", y_prednew2[0])
```


## Output:
```
Original Data:
   sl_no gender  ssc_p    ssc_b  hsc_p    hsc_b     hsc_s  degree_p  \
0      1      M  67.00   Others  91.00   Others  Commerce     58.00   
1      2      M  79.33  Central  78.33   Others   Science     77.48   
2      3      M  65.00  Central  68.00  Central      Arts     64.00   
3      4      M  56.00  Central  52.00  Central   Science     52.00   
4      5      M  85.80  Central  73.60  Central  Commerce     73.30   

    degree_t workex  etest_p specialisation  mba_p      status    salary  
0   Sci&Tech     No     55.0         Mkt&HR  58.80      Placed  270000.0  
1   Sci&Tech    Yes     86.5        Mkt&Fin  66.28      Placed  200000.0  
2  Comm&Mgmt     No     75.0        Mkt&Fin  57.80      Placed  250000.0  
3   Sci&Tech     No     66.0         Mkt&HR  59.43  Not Placed       NaN  
4  Comm&Mgmt     No     96.8        Mkt&Fin  55.50      Placed  425000.0  

After dropping 'sl_no' and 'salary':
  gender  ssc_p    ssc_b  hsc_p    hsc_b     hsc_s  degree_p   degree_t  \
0      M  67.00   Others  91.00   Others  Commerce     58.00   Sci&Tech   
1      M  79.33  Central  78.33   Others   Science     77.48   Sci&Tech   
2      M  65.00  Central  68.00  Central      Arts     64.00  Comm&Mgmt   
3      M  56.00  Central  52.00  Central   Science     52.00   Sci&Tech   
4      M  85.80  Central  73.60  Central  Commerce     73.30  Comm&Mgmt   

  workex  etest_p specialisation  mba_p      status  
0     No     55.0         Mkt&HR  58.80      Placed  
1    Yes     86.5        Mkt&Fin  66.28      Placed  
2     No     75.0        Mkt&Fin  57.80      Placed  
3     No     66.0         Mkt&HR  59.43  Not Placed  
4     No     96.8        Mkt&Fin  55.50      Placed  

Data types after converting to 'category':
gender            category
ssc_p              float64
ssc_b             category
hsc_p              float64
hsc_b             category
hsc_s             category
degree_p           float64
degree_t          category
workex            category
etest_p            float64
specialisation    category
mba_p              float64
status            category
dtype: object

Data after converting categories to numeric codes:
   gender  ssc_p  ssc_b  hsc_p  hsc_b  hsc_s  degree_p  degree_t  workex  \
0       1  67.00      1  91.00      1      1     58.00         2       0   
1       1  79.33      0  78.33      1      2     77.48         2       1   
2       1  65.00      0  68.00      0      0     64.00         0       0   
3       1  56.00      0  52.00      0      2     52.00         2       0   
4       1  85.80      0  73.60      0      1     73.30         0       0   

   etest_p  specialisation  mba_p  status  
0     55.0               1  58.80       1  
1     86.5               0  66.28       1  
2     75.0               0  57.80       1  
3     66.0               1  59.43       0  
4     96.8               0  55.50       1  

Feature matrix X shape: (215, 12)
Target vector y shape: (215,)

Initial theta (weights):
[-0.67527549 -0.77872176  1.28568344  1.29374691  2.55427021  1.03847811
 -0.57659603  0.27116525 -0.75934491 -1.45128292  0.85393583  0.68748657]
Iteration 100, Loss: 1523.7301
Iteration 200, Loss: 732.9008
Iteration 300, Loss: 703.5376
Iteration 400, Loss: 1113.4401
Iteration 500, Loss: 405.5158
Iteration 600, Loss: 427.0418
Iteration 700, Loss: 453.7732
Iteration 800, Loss: 354.9237
Iteration 900, Loss: 334.4840
Iteration 1000, Loss: 670.3141

Final theta (weights) after training:
[-0.63185983  1.72476707  1.30008531  0.84472264  2.46405078  0.83756068
  0.35374623  0.08512123 -0.32261219 -0.15593487  0.41723809 -2.51516578]

Training Accuracy: 0.7767441860465116
Predicted labels (first 20):
[1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 0 1 0]

Prediction for new student 1 (0 = Not Placed, 1 = Placed): 1
Prediction for new student 2 (0 = Not Placed, 1 = Placed): 1
```


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
