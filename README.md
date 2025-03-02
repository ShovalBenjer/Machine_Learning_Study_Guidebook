# Machine Learning Comprehensive Study Guide

## Introduction

This comprehensive guide covers key topics in machine learning—from fundamental concepts to advanced techniques. It's designed to serve as both a reference and a learning resource for students, practitioners, and enthusiasts at various stages of their machine learning journey.

## Table of Contents

- [Supervised Learning](#supervised-learning)
  - [Overview](#supervised-learning-overview)
  - [Training, Validation, and Testing](#training-validation-and-testing)
  - [Linear Regression](#linear-regression)
  - [Loss Functions](#loss-functions)
  - [Gradient Descent](#gradient-descent)
  - [Stochastic Gradient Descent](#stochastic-gradient-descent)
  - [Feature Normalization](#feature-normalization)
  - [Classification](#classification)
  - [Logistic Regression](#logistic-regression)
  - [Cross Entropy](#cross-entropy)
  - [Multi-Category Classification](#multi-category-classification)
  - [One vs. One Multi-Class Classification](#one-vs-one-multi-class-classification)
  - [Multiclass Classification Extensions](#multiclass-classification-extensions)
- [Model Evaluation](#model-evaluation)
  - [Bias vs. Variance Trade-off](#bias-vs-variance-trade-off)
  - [Key Performance Indicators (KPIs)](#key-performance-indicators-kpis)
  - [Feature Selection](#feature-selection)
  - [Validation & Cross-Validation](#validation--cross-validation)
  - [Data Preprocessing, Feature Engineering & Encoding](#data-preprocessing-feature-engineering--encoding)
- [Fighting High Variance (Overfitting)](#fighting-high-variance-overfitting)
- [Advanced Classification Methods](#advanced-classification-methods)
  - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
  - [Bayesian Classifiers](#bayesian-classifiers)
  - [Decision Trees & Random Forests](#decision-trees--random-forests)
- [Unsupervised Learning](#unsupervised-learning)
  - [Overview](#unsupervised-learning-overview)
  - [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
  - [K-Means Clustering](#k-means-clustering)
  - [Hierarchical Clustering](#hierarchical-clustering)
- [Text Feature Extraction](#text-feature-extraction)
- [Practical Tools & Resources](#practical-tools--resources)

## Supervised Learning

### Supervised Learning Overview

Supervised learning is a field in machine learning where we have data pairs that include inputs and their corresponding target outputs. The goal is to find a model that maps from inputs to outputs.

Learning is based on labeled examples, and we use an algorithm to predict outputs for new, unseen data.

#### Stages:
1. **Exploration** - Examining the relationships between features and understanding the nature of the data.
2. **Model Creation** - Building models that can map inputs to their target outputs.
3. **Model Evaluation** - Testing the performance of the model on unseen data.
4. **Synthesis** - Combining steps 1-3 to develop a unified methodology that addresses the problem in the best way possible.

#### Features
- **Numerical features**: Features that represent numerical values.
- **Categorical features**: Features that represent categories or groups.

### Training, Validation, and Testing

In supervised learning, where y = h(x), we're trying to find a function h that can map input x to output y.

The function takes input x and parameters that need to be learned from the data. By finding optimal parameters, we can make the model predict outputs with high accuracy, minimizing the prediction error.

#### Stages:
1. **Training** - Using the training set to learn the model parameters.
2. **Validation** - Using a separate set to tune hyperparameters and make sure we're not overfitting.
3. **Final Test** - Using a third, completely unseen dataset to evaluate the model's performance.

## Linear Regression

### Overview
Linear Regression is a supervised learning technique for modeling the relationship between a numeric target and one or more features by fitting a linear equation. It's one of the most fundamental and widely used algorithms in machine learning.

### Model Formulation 
For a single variable, the model is:

$$\hat{y} = w_0 + w_1 x$$

In the multivariable case, it is:

$$\hat{y} = w_0 + w_1 x_1 + \dots + w_p x_p$$

Where:
- $w_0$ is the intercept (bias)
- $w_1, ..., w_p$ are the weights/coefficients
- $x_1, ..., x_p$ are the input feature values

Given a dataset D = {(x_i, t_i)}, we want to find h(x) that approximates the true function.

![Linear Regression Fit](https://upload.wikimedia.org/wikipedia/commons/e/ed/Residuals_for_Linear_Regression_Fit.png)  
*Figure: A scatterplot with a red best-fit line and residuals shown as vertical lines.*

### Key Concepts
- **Linear Hypothesis:** The assumption of a linear relationship between features and target.
- **Parameters/Weights:**  
  - **Intercept/Bias:** The predicted value when all features are zero.  
  - **Coefficients:** The slopes showing how each feature affects the prediction.
- **Training vs. Test Error:** Training error is measured on the training data; generalization (test) error shows performance on unseen data.
- **Outliers:** Outliers can disproportionately affect the model, skewing the results.

The goal is to find the parameters (w₀, w₁, ..., w_n) that minimize the error between predictions and actual values in dataset D.

[![An Intuitive Introduction to Linear Regression](https://img.youtube.com/vi/3g-e2aiRfbU/maxresdefault.jpg)](https://youtu.be/3g-e2aiRfbU)

### Loss Functions

The loss function measures the model's prediction error. Common loss functions include:

1) **Sum of Squared Errors (SSE)**: 
   $$\sum(t_i - h(x_i))^2$$

2) **Mean Squared Error (MSE)**: 
   $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i-\hat{y}_i)^2$$

3) **Root Mean Squared Error (RMSE)**: 
   $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i-\hat{y}_i)^2}$$

4) **Mean Absolute Error (MAE):**  
   $$\text{MAE} = \frac{1}{n}\sum |y - \hat{y}|$$

5) **R-squared:** Proportion of variance in the target explained by the model.

### Gradient Descent

Gradient Descent is an optimization algorithm used to minimize the loss function by iteratively adjusting model parameters.

#### Steps:
1. Initialize the parameters (w⃗, w₀) randomly.
2. Calculate the gradient of the loss function with respect to the parameters.
3. Update the parameters (w_i) in the opposite direction of the gradient to minimize the loss.

The update rule is:
$$\Delta w_i = -\lambda \cdot \frac{\partial \text{Loss}(w)}{\partial w_i}$$

Where:
- $\lambda$ is the learning rate
- $\frac{\partial \text{Loss}(w)}{\partial w_i}$ is the partial derivative of the loss function with respect to w_i.

GD is an off-line algorithm, meaning we process all data before making an update. This makes it computationally expensive with large datasets.

### Stochastic Gradient Descent

In contrast to regular Gradient Descent, Stochastic Gradient Descent (SGD) processes data in mini-batches, making it more efficient for large datasets.

#### Steps:
1. Initialize parameters (w⃗, w₀).
2. Shuffle and divide the dataset into m mini-batches.
3. For each mini-batch, calculate h(x) and update the parameters.
4. Repeat for multiple epochs (passes through the data).

The update rule for SGD is:
$$\Delta w_i = -\frac{\lambda}{m} \cdot \sum(t_i - y_i)x_i$$

Where:
- m is the mini-batch size

SGD is more efficient than GD for large datasets because it updates parameters more frequently, potentially leading to faster convergence.

The gradient computation for cross entropy loss in SGD:
$$\Delta w_i = -\frac{\lambda}{m} \cdot \sum\left(\frac{\partial \text{Loss}(t_i,y_i)}{\partial w_i}\right)$$

### Feature Normalization

Feature normalization is critical for gradient descent to work efficiently, especially when features have different scales. It helps the algorithm converge faster and prevents features with larger scales from dominating.

#### Methods:
1) **Min-Max Scaling**:
   $$x_{\text{normalized}} = \frac{x - \min(x)}{\max(x) - \min(x)}$$

2) **Mean Normalization**:
   $$x_{\text{normalized}} = x - \text{mean}(x)$$

3) **Standard Deviation Scaling**:
   $$x_{\text{normalized}} = \frac{x}{\text{SD}(x)}$$

4) **Standard Normalization (Z-score)**:
   $$x_{\text{normalized}} = \frac{x - \text{mean}(x)}{\text{SD}(x)}$$
   
   This centers the data around 0 with a standard deviation of 1.

## Classification

### Classification Overview

The goal of classification is to learn a function h(x⃗) that minimizes the loss function for dataset D = {(x_i, t_i)}, where t_i represents the class label and y = P(t=1|x) is the probability of class 1 given input x.

The key difference between classification and regression is that classification predicts categorical outputs rather than continuous values.

### Logistic Regression

#### Overview
Logistic Regression is used for binary classification. It models the probability of a positive class by applying the sigmoid function to a linear combination of features:

$$p(x) = \frac{1}{1 + e^{-(w_0 + w_1 x_1 + \dots + w_p x_p)}}$$

A probability above a chosen threshold (typically 0.5) indicates the positive class.

The function is: 
$$y = g(z) = \frac{1}{1+e^{-z}}$$

where $z = w₀ + \sum w_i x_i$

The function g is the sigmoid (logistic) function that maps any input to a value between 0 and 1, representing the probability of the positive class.

![Logistic Regression Curve](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/Exam_pass_logistic_curve.svg/1280px-Exam_pass_logistic_curve.svg.png)  
*Figure: The sigmoid function mapping inputs to probabilities.*

#### Key Concepts
- **Sigmoid Function:** Transforms any real value into a value between 0 and 1.
- **Binary Classification:** Predicts one of two classes.
- **Odds Ratio:** Exponentiating coefficients gives the factor change in odds for a unit change in a feature.
- **Maximum Likelihood Estimation:** Training method to optimize the model parameters.

**Advantage**: Provides a probability estimate, not just a classification.
**Disadvantage**: May struggle with highly non-linear decision boundaries.

The decision boundary is where w⃗·x⃗ = 0:
$$\frac{w⃗·x⃗}{||w⃗||} = \frac{1}{||w⃗||} \cdot d(x⃗, w⃗)$$

Where d is the distance from point x⃗ to the decision boundary.

[![An Quick Intro to Logistic Regression](https://img.youtube.com/vi/EKm0spFxFG4/maxresdefault.jpg)](https://youtu.be/EKm0spFxFG4)

### Cross Entropy

Cross Entropy (CE) is the loss function used for logistic regression:

For a pair (x,t):
$$C(y,t) = -t \cdot \log(y) - (1-t) \cdot \log(1-y) = 
\begin{cases}
-\log(y) & \text{if } t=1 \\
-\log(1-y) & \text{if } t=0
\end{cases}$$

The total CE loss for a dataset with m examples:
$$\text{CE}_D(w) = \frac{1}{m} \cdot \sum C(y_i, t_i)$$

The perplexity is defined as:
$$\text{Perplexity} = e^{\text{CE}_D(w)}$$

### Gradient Descent with Cross Entropy

The gradient of the cross entropy with respect to the weights:
$$\frac{\partial \text{CE}_D(t,y)}{\partial w_i} = \frac{\partial \text{CE}_D(t,y)}{\partial y} \cdot \frac{\partial y}{\partial w_i} = \frac{1}{m} \cdot \sum(y_i - t_i)x_i$$

The update rule becomes:
$$\Delta w_i = -\frac{\lambda}{m} \cdot \sum(t_i - y_i)x_i$$

#### Comparison of GD and SGD with Cross Entropy:

- **GD**: $\Delta w_i = -\frac{\lambda}{m} \cdot \sum(t_i - y_i)x_i$, where m = |D| (full dataset)
- **SGD**: $\Delta w_i = -\frac{\lambda}{m} \cdot \sum(t_i - y_i)x_i$, where m = |minibatch| (subset of data)

**Advantages of GD**: The CE loss calculation is more stable, which leads to consistent updates.
**Disadvantages of GD**: Takes longer to process all the data, which can be inefficient with large datasets.

### Multi-Category Classification

Extending binary classification to handle multiple categories:

#### One vs. Rest (One vs. All):
- Train n binary classifiers, where n is the number of classes
- Each classifier distinguishes one class from all others
   
**Advantage**: Simple to implement and extends binary classifiers naturally.
**Disadvantage**: Class imbalance issues as the negative examples often outnumber the positive ones.

### One vs. One Multi-Class Classification

In this approach, we train n(n-1)/2 binary classifiers, each one trained on a pair of classes from the original dataset.

There are several approaches to combine the results from these classifiers:

#### 1. **Majority Vote**
Choose the class that wins the most pairwise comparisons.

#### 2. **Probabilistic Combination**
Combine the probability estimates from all n-1 classifiers that involve the specific class. Each classifier contributes to the probability estimation of the classes it was trained to distinguish.

#### 3. **Decision-Based Weighting**
Weight each classifier's decision based on its confidence or its performance on the validation set.

### Multiclass Classification Extensions

#### MCCE (Multiclass Cross-Entropy)
Used for native multiclass classification problems.

#### Softmax + CCE (Categorical Cross-Entropy)
The softmax function converts raw model outputs into probabilities across all classes, and CCE measures the difference between these probabilities and the true distribution.

#### Unified Classifier CE
A single model that directly produces probabilities for all classes, trained with cross-entropy loss.

## Model Evaluation

### Bias vs. Variance Trade-off

The Bias-Variance Trade-off describes the balance between the error from erroneous assumptions (bias) and the error from sensitivity to small data fluctuations (variance). An optimal model minimizes overall error by finding a balance between underfitting and overfitting.

$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

High-bias models are typically simpler with limited flexibility, which may lead to underfitting. They struggle to capture complex patterns in the data and have high error on both training and test sets.

Underfitting occurs when the model is too simple relative to the underlying data complexity. When this happens, the model has high bias but low variance.

Low-bias, high-variance models are more complex and flexible, potentially leading to overfitting.

![Bias–Variance Trade-off](https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Bias_and_variance_contributing_to_total_error.svg/1024px-Bias_and_variance_contributing_to_total_error.svg.png)  
*Figure: Illustration of bias and variance contributions to total error.*

### Key Performance Indicators (KPIs)

Key Performance Indicators (KPIs) are metrics used to evaluate classification and regression models.

#### Classification Metrics:

**Confusion Matrix:**  

|                      | Predicted Positive | Predicted Negative |
|----------------------|--------------------|--------------------|
| **Actual Positive**  | True Positive (TP) | False Negative (FN)|
| **Actual Negative**  | False Positive (FP)| True Negative (TN) |

**Accuracy:**  
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precision:**  
$$\text{Precision} = \frac{TP}{TP+FP}$$

**Recall (Sensitivity):**  
$$\text{Recall} = \frac{TP}{TP+FN}$$

**F1 Score:**  
$$F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Specificity:**  
$$\text{Specificity} = \frac{TN}{TN+FP}$$

**ROC AUC:** The area under the Receiver Operating Characteristic curve, summarizing the trade-off between sensitivity and specificity.

#### Regression Metrics:

**Mean Squared Error (MSE):**  
$$\text{MSE} = \frac{1}{n}\sum (y - \hat{y})^2$$

**Root Mean Squared Error (RMSE):**  
$$\text{RMSE} = \sqrt{\text{MSE}}$$

**Mean Absolute Error (MAE):**  
$$\text{MAE} = \frac{1}{n}\sum |y - \hat{y}|$$

**R-squared:** Proportion of variance in the target explained by the model.

### Feature Selection

When addressing high-bias/underfitting issues, consider:

#### 1. **Increasing Feature Complexity**
Adding more complex features or transformations of existing features.

#### 2. **Adding Interaction Features**
Creating new features that capture interactions between existing features.

When dealing with many features, it's important to rank them by importance. Features that contribute more to reducing the error are considered more important. This can be determined by statistical methods or by measuring the impact of each feature on the model's performance.

### Validation & Cross-Validation

Validation is critical to ensure a model's generalizability. Data is typically divided into training, validation, and test sets. Cross-validation, such as k-fold cross-validation, systematically rotates the validation set to produce a robust estimate of model performance without overfitting to a single split.

![K-Fold Cross-Validation](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/K-fold_cross_validation_EN.svg/1024px-K-fold_cross_validation_EN.svg.png)  
*Figure: Illustration of k-fold cross-validation where the dataset is partitioned into k subsets.*

#### Key Concepts:
- **Training Set:** Data used to learn model parameters.
- **Validation Set:** Data used for tuning hyperparameters and model selection.
- **Test Set:** A final hold-out set used only for final performance evaluation.
- **Hold-Out Validation:** A single split of data into training and test sets.
- **k-Fold Cross-Validation:** The data is divided into k parts and the model is trained and validated k times.
- **Hyperparameter Tuning:** Adjusting model settings (e.g., learning rate, tree depth) based on validation performance.

[![Cross Validation](https://img.youtube.com/vi/fSytzGwwBVw/maxresdefault.jpg)](https://youtu.be/fSytzGwwBVw)

### Data Preprocessing, Feature Engineering & Encoding

#### Data Cleansing:  
Data cleansing involves detecting and correcting errors in the dataset—handling missing values, removing or correcting outliers, and standardizing formats. This step ensures the data is clean and consistent, which is crucial for building reliable models.

#### Feature Creation (Engineering):  
Feature engineering is the process of transforming raw data into informative features that improve model performance.  
- *Examples:* Creating polynomial features, extracting date components, or aggregating data within groups.  
- *Impact:* Good features can greatly improve the model's ability to learn patterns; poor features can lead to underfitting or overfitting.

#### One-Hot Encoding:  
One-hot encoding converts categorical variables into a set of binary features, each indicating the presence (1) or absence (0) of a category. This encoding is essential for algorithms that require numerical input and avoids implying an ordinal relationship between categories.

#### Train–Validation–Test Split:  
Properly splitting the dataset ensures that model performance is evaluated on unseen data. The typical approach involves:
- **Training Set:** For fitting the model.
- **Validation Set:** For tuning hyperparameters.
- **Test Set:** For final performance evaluation.

Using cross-validation can further refine performance estimates when data is limited.

## Fighting High Variance (Overfitting)

High variance (overfitting) occurs when a model captures noise in the training data. Strategies to fight high variance include:

- **Regularization:** Techniques like L1 (Lasso) and L2 (Ridge) add penalty terms to the loss function to discourage overly complex models.
- **Ensemble Methods:** Combining multiple models (e.g., bagging, boosting) to reduce overall variance.
- **Early Stopping:** Monitoring validation performance to halt training before overfitting occurs.
- **Feature Selection/Dimensionality Reduction:** Reducing the number of features can help prevent overfitting.
- **Increasing Training Data:** More data generally leads to better generalization.

[![L1 Vs L2 Regularzation Methods](https://img.youtube.com/vi/aBgMRXSqd04/maxresdefault.jpg)](https://youtu.be/aBgMRXSqd04)

## Unsupervised Classification Methods

### K-Nearest Neighbors (KNN)

KNN is an instance-based learning algorithm for both classification and regression. It predicts new examples by finding the *k* nearest training examples using a distance metric.

#### Key Points:

- **Distance Metrics:**  
  - Euclidean Distance, Manhattan Distance, Cosine Similarity.
- **Parameter \(k\):** Determines the balance between bias and variance.
- **Normalization:** Essential to scale features.
- **Curse of Dimensionality:** In high dimensions, distance measures may become less meaningful.

[![K nearest Neighbors](https://img.youtube.com/vi/b6uHw7QW_n4/maxresdefault.jpg)](https://youtu.be/b6uHw7QW_n4)

### Bayesian Classifiers

Bayesian classifiers use Bayes' Theorem to compute class probabilities given features. The Naive Bayes classifier assumes feature independence, making it simple yet effective—especially in text classification.

$$P(C \mid X) \propto P(C) \prod_{i=1}^n P(X_i \mid C)$$

#### Key Points:

- **Naive Bayes:** Fast and effective despite the naive assumption.
- **Laplace Smoothing:** Prevents zero probabilities for unseen feature values.
- **Text Classification:** Often uses a "bag-of-words" approach.

### Decision Trees & Random Forests

Decision Trees split data based on feature tests to predict outcomes, forming a tree-like structure. Random Forests aggregate multiple decision trees built on random subsets of data and features, reducing variance and improving robustness.

#### Key Points:

- **Decision Trees:**  
  - Split data using criteria like information gain or Gini impurity.
  - Highly interpretable.

![Decision Tree Example](https://github.com/user-attachments/assets/ffada2e4-eeac-4c18-945d-49abb7118930)
  
*Figure: A decision tree example for the Iris dataset.*

- **Pruning:**  
  - Prevents overfitting by trimming unimportant branches.
  - [![Bagging vs Boosting](https://github.com/user-attachments/assets/c600802c-82b6-4b7a-818f-e74e1e427be1)](https://youtu.be/tjy0yL1rRRU)

- **Random Forests:**  
  - Use bagging and random feature selection.
  - Provide feature importance measures.

![Random Forest Illustration](https://github.com/user-attachments/assets/032b91d7-7bb0-4978-8973-e6edbe75c89c)

*Figure: Multiple decision trees in a Random Forest.*

## Unsupervised Learning

### Unsupervised Learning Overview

Unsupervised learning deals with datasets that don't have labeled outputs. The goal is to discover patterns, structures, or relationships within the data.

### Principal Component Analysis (PCA)

PCA is a dimensionality reduction technique that identifies the directions (principal components) along which the data varies the most. It's useful for:
- Reducing the number of features while preserving most of the information
- Visualizing high-dimensional data
- Eliminating correlated features

#### Process:
- Standardize data
- Compute covariance matrix
- Derive eigenvalues/eigenvectors
- Select top components

#### Usage:
- Visualization
- Speeding up models
- Mitigating multicollinearity

### K-Means Clustering

K-Means partitions data into *K* clusters by iteratively assigning points to the nearest centroid and updating centroids.

#### Key Concepts:
- **Choosing K:** Techniques include the elbow method and silhouette score.
- **Limitations:** Assumes spherical clusters and can be sensitive to initialization.

### Hierarchical Clustering

Hierarchical clustering builds a tree (dendrogram) of clusters without pre-specifying *K*.

#### Key Approaches:
- **Agglomerative Approach:** Start with individual points, merge clusters iteratively.
- **Linkage Criteria:** Single, complete, average, or Ward's method determine merging criteria.

## Text Feature Extraction

### N-Gram Features

N-grams are contiguous sequences of *n* items (words or characters) from text.
- **Usage:** Capturing context in text data; e.g., unigrams, bigrams, trigrams.
- **Trade-Off:** Higher n-grams provide more context but increase feature space dimensionality.

### TF-IDF (Term Frequency–Inverse Document Frequency)

TF-IDF weights text features by emphasizing terms that are frequent in a document but rare across documents.
- **TF:** Frequency of a term in a document.
- **IDF:** Logarithmic measure that downweights common terms.
- **Application:** Enhances text classification and search by highlighting distinctive words.

---

# Further Reading

### Sklearn Documentation

![image](https://github.com/user-attachments/assets/a41a4bff-325d-44e6-9dd9-6cc862d94057)  

[Sklearn Documentation link - Press here](https://scikit-learn.org/stable/supervised_learning.html)

## Practical Tools & Resources

### Kaggle
A platform for data science competitions, datasets, and collaborative learning.
- **Competitions:** Real-world problems with leaderboards.
- **Datasets:** Diverse and extensive collections for practice.
- **Notebooks & Forums:** Community sharing of code and strategies.

### PyPI
- **PyPI:** The Python Package Index, hosting libraries like scikit-learn, TensorFlow, and more.

### Plotly
An interactive plotting library for creating dynamic, publication-quality visualizations.
- **Usage:** Ideal for interactive data exploration in notebooks and web dashboards.
- **Interfaces:** High-level (plotly.express) and low-level (graph_objects).

### Automated EDA Tools
- **AutoViz:** Automatically generates visualizations from datasets, providing quick insights.
- **D-Tale:** An interactive GUI for exploring and editing Pandas DataFrames in a web browser.

## Glossary

### General Machine Learning Terms
- **Supervised Learning:** Learning from labeled data with target outputs.
- **Unsupervised Learning:** Learning from data without labeled outputs.
- **Feature:** An individual measurable property of the data.
- **Label/Target:** The output variable to be predicted.
- **Training Set:** Data used to train the model.
- **Validation Set:** Data used to tune hyperparameters.
- **Test Set:** Data used for final evaluation.
- **Overfitting:** When a model performs well on training data but poorly on new data.
- **Underfitting:** When a model is too simple to capture the underlying pattern.

### Linear and Logistic Regression
- **Dependent variable (Target):** The output \(y\) the model aims to predict.
- **Independent variable (Feature):** The input \(x\) used for prediction.
- **Coefficient (Weight):** Parameter \(w_i\) indicating the effect of a feature on \(y\).
- **Intercept (Bias):** Constant \(w_0\) representing the output when all features are zero.
- **Least Squares:** Method to fit the model by minimizing the sum of squared residuals.
- **Residual (Error):** The difference \(y - \hat{y}\).
- **Sigmoid/Logistic Function:** Function mapping linear outputs to probabilities.
- **Odds & Log-Odds:** Odds are given by \(\frac{p}{1-p}\) and log-odds are the natural logarithm of the odds.
- **Binary Classification:** Predicting two possible outcomes.
- **Decision Threshold:** The cutoff (often 0.5) used to classify the probability output.
- **Odds Ratio:** \(e^{w_i}\); quantifies the change in odds per unit increase in feature \(i\).
- **Multinomial Logistic Regression:** Extends logistic regression to multi-class problems.

### Model Evaluation
- **True Positive/Negative (TP/TN):** Correctly predicted instances.
- **False Positive/Negative (FP/FN):** Misclassified instances.
- **Precision:** The proportion of positive predictions that are correct.
- **Recall:** The proportion of actual positives that are correctly predicted.
- **F1 Score:** Harmonic mean of precision and recall.
- **Specificity:** True negative rate.
- **ROC AUC:** A summary measure of a classifier's ability to distinguish between classes.
- **MSE, RMSE, MAE:** Different ways to quantify error in regression.
- **R-squared:** The fraction of variance in the target explained by the model.
- **Bias:** Error due to simplifying assumptions; high bias leads to underfitting.
- **Variance:** Error due to sensitivity to training data; high variance leads to overfitting.
- **Model Complexity:** Degree of flexibility in a model.
- **Regularization:** Methods to constrain a model, reducing variance.

### Advanced Methods
- **Instance-Based Learning:** Learning deferred until a query is made.
- **Distance Metric:** A function to measure similarity.
- **Weighted KNN:** Giving more influence to nearer neighbors.
- **Curse of Dimensionality:** Degradation of distance measures in high dimensions.
- **Bayes' Theorem:** \(P(A|B)=\frac{P(B|A)P(A)}{P(B)}\)
- **Prior Probability:** \(P(C)\)
- **Likelihood:** \(P(X|C)\)
- **Posterior Probability:** \(P(C|X)\)
- **Conditional Independence:** Assumption that features are independent given the class.
- **Laplace Smoothing:** Technique to handle zero counts.
- **Decision Tree:** Model that uses a series of splits.
- **Node/Leaf/Branch:** Parts of a tree structure.
- **Impurity Measures:** Metrics like Gini or entropy.
- **Information Gain:** Reduction in impurity from a split.
- **Pruning:** Process to avoid overfitting.
- **Bagging:** Bootstrap aggregating.
  [![ Bootstrap Aggregating - Bagging](https://github.com/user-attachments/assets/7400fa6e-54d3-4da2-b3cd-88ab283d6ec5)](https://www.youtube.com/watch?v=2Mg8QD0F1dQ)
- **Random Forest:** Ensemble of decision trees.
- **Feature Importance:** Contribution measure of a feature.
- **Out-of-Bag Error:** Internal error estimate.
- **Regularization:** Techniques to constrain the model's complexity.
- **Lasso (L1) and Ridge (L2):** Methods to shrink coefficients, with Lasso also performing feature selection
