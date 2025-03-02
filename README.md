# Machine Learning Comprehensive Study Guide

## Introduction

This comprehensive guide covers key topics in machine learning—from fundamental concepts to advanced techniques. It's designed to serve as both a reference and a learning resource for students, practitioners, and enthusiasts at various stages of their machine learning journey.

This study guide includes concepts from computational learning and statistical learning perspectives, aiming to provide a holistic view of machine learning.

## Table of Contents

- [Supervised Learning](#supervised-learning)
  - [Overview](#supervised-learning-overview)
  - [Training, Validation, and Testing](#training-validation-and-testing)
  - [Linear Regression](#linear-regression)
    - [Overview](#linear-regression-overview)
    - [Mathematical Formulation](#mathematical-formulation-linear-regression)
    - [Loss Function](#loss-function-linear-regression)
    - [Error Metrics](#error-metrics-linear-regression)
    - [Ridge and Lasso Regression & Regularization](#ridge-and-lasso-regression--regularization)
  - [Loss Functions](#loss-functions)
  - [Gradient Descent](#gradient-descent)
  - [Stochastic Gradient Descent](#stochastic-gradient-descent)
  - [Feature Normalization](#feature-normalization)
  - [Classification](#classification)
  - [Logistic Regression](#logistic-regression)
    - [Overview](#logistic-regression-overview)
    - [Mathematical Formulation](#mathematical-formulation-logistic-regression)
    - [Loss Function](#loss-function-logistic-regression)
    - [Gradient Descent](#gradient-descent-logistic-regression)
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
    - [Overview](#knn-overview)
    - [Algorithm](#knn-algorithm)
  - [Bayesian Classifiers](#bayesian-classifiers)
  - [Decision Trees & Random Forests](#decision-trees--random-forests)
  - [Support Vector Machines (SVM)](#support-vector-machines-svm)
    - [Overview](#svm-overview)
    - [Kernels](#kernels-svm)
- [Unsupervised Learning](#unsupervised-learning)
  - [Overview](#unsupervised-learning-overview)
  - [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
    - [Overview](#pca-overview)
    - [Mathematical Formulation](#mathematical-formulation-pca)
  - [K-Means Clustering](#k-means-clustering)
  - [Hierarchical Clustering](#hierarchical-clustering)
- [Linear Algebra Fundamentals](#linear-algebra-fundamentals)
  - [Inner/Scalar Product](#inner-scalar-product)
  - [L-norms](#l-norms)
  - [Metric Distance](#metric-distance)
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

### Linear Regression Overview

Linear Regression is a supervised learning technique for modeling the relationship between a numeric target and one or more features by fitting a linear equation. It's one of the most fundamental and widely used algorithms in machine learning.

From a computational learning perspective, linear regression is a method that predicts features based on a large database.

### Mathematical Formulation (Linear Regression)
We input two types of data:

1. **Feature Matrix** $X$: An $n \times d$ matrix where each row is a sample ($n$ samples) and each column represents features ($d$ features).
2. **Observation Vector** $y$: A column vector of size $n$ containing the observations/results for each sample.

In linear regression, we aim to find a straight line that best describes the relationship between the samples $X$ and observations $y$. The equation is:
$y = \omega_0 + X \cdot \hat{\omega}$

Because linear regression is used for stochastic data (data without a single mathematical relationship, with noise depending on the samples we took), the equation includes an error term:
$y = \hat{y} + \varepsilon$

Here:
- $y$ is the actual result,
- $\varepsilon$ represents the error (normally distributed with mean 0), and
- $\hat{y}$ is our prediction based on a specific input.

**Geometrical Illustration:** [PLACEHOLDER FOR GEOMETRICAL ILLUSTRATION: Shows $\hat{y}$ as the projection of $y$ onto the space spanned by features $X$]

Geometrically, $\hat{y}$ can be visualized as the projection of $y$ onto the space spanned by the features $X$. For instance, with 2 features, $y$ would be in a 3-dimensional space, and $\hat{y}$ would be its projection onto the plane spanned by those two features.

To incorporate the intercept term $\omega_0$ into matrix operations, we use an augmented feature matrix $\tilde{X}$. This matrix is the same as $X$ but with an added column of ones as the first column, ensuring $\omega_0$ is not multiplied by any feature:
$\hat{y} = \tilde{X} \cdot \hat{\omega}$

### Model Formulation (Linear Regression)
For a single variable, the model is:
$\hat{y} = w_0 + w_1 x$

In the multivariable case:
$\hat{y} = w_0 + w_1 x_1 + \dots + w_p x_p$

Where:
- $w_0$ is the intercept (bias)
- $w_1, ..., w_p$ are the weights/coefficients
- $x_1, ..., x_p$ are the input feature values

Given a dataset D = {(x_i, t_i)}, our goal is to find a function h(x) that closely approximates the true function.

![Linear Regression Fit](https://upload.wikimedia.org/wikipedia/commons/e/ed/Residuals_for_Linear_Regression_Fit.png)
*Figure: A scatterplot showing a red best-fit line with residuals as vertical lines.*

### Key Concepts (Linear Regression)

- **Linear Hypothesis:** The assumption that the relationship between features and the target is linear.
- **Parameters/Weights:**
  - **Intercept/Bias ($w_0$):** The predicted value when all features are zero.
  - **Coefficients ($w_1, ..., w_p$):** Slopes indicating how each feature influences the prediction.
- **Training vs. Test Error:** Training error is measured on the data used for training, while generalization (test) error reflects performance on unseen data.
- **Outliers:** Data points that can disproportionately influence the model, potentially skewing results.

The objective is to determine the parameters ($w_0, w_1, ..., w_n$) or $\hat{\omega}$ that minimize the discrepancy between the model's predictions and the actual values in dataset D.

[![An Intuitive Introduction to Linear Regression](https://img.youtube.com/vi/3g-e2aiRfbU/maxresdefault.jpg)](https://youtu.be/3g-e2aiRfbU)

### Loss Function (Linear Regression)

To find the optimal model, we need to define a **loss function** that quantifies the error of our predictions. We aim to find the parameters $\omega$ that minimize this loss. A common loss function for linear regression is the **Sum of Squared Errors (SSE)**, also known as the **LOSS** function in this context:

#### Loss Function
$LOSS = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$

Other related loss and error functions include:

1) **Sum of Squared Errors (SSE)**:
   $\sum(t_i - h(x_i))^2$

2) **Mean Squared Error (MSE)**:
   $\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i-\hat{y}_i)^2$

3) **Root Mean Squared Error (RMSE)**:
   $\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i-\hat{y}_i)^2}$

4) **Mean Absolute Error (MAE):**
   $\text{MAE} = \frac{1}{n}\sum |y - \hat{y}|$

5) **R-squared:** Proportion of variance in the target explained by the model.

The analytical solution that provides the optimal weight vector $\hat{w}$ minimizing the LOSS function is given by the **Normal Equation**:
$\hat{w} = (\tilde{X}^T\tilde{X})^{-1} \tilde{X}^Ty$

Consequently, the predicted values $\hat{y}$ are:
$\hat{y} = \tilde{X}(\tilde{X}^T\tilde{X})^{-1}\tilde{X}^T \cdot y$

### Error Metrics (Linear Regression)

To evaluate the performance of our linear regression model, especially on test data ($x_{test}$), we use several error metrics:

- **Residual Sum of Squares (RSS)**: Measures the variance of the residuals (prediction errors).
  $\text{RSS} = \sum_{i=1}^{n} (\hat{y}_i - \bar{y})^2$

- **Total Sum of Squares (TSS)**: Represents the total variance in the dependent variable.
  $\text{TSS} = \sum_{i=1}^{n} (y_i - \bar{y})^2 \rightarrow \text{Variance}$

- **Error Sum of Squares (ESS)**: Measures the sum of the squared differences between the actual and predicted values.
  $\text{ESS} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$

  Where $\bar{y} = \frac{\sum y_i}{N}$ is the mean of the observed values $y_i$.

- **Mean Square Error (MSE)**: The average of the squared errors.
  $\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$

- **Mean Square Residual (MSR)**: The RSS normalized by the number of features.
  $\text{MSR} = \frac{\text{RSS}}{d}$

  Where $n$ is the number of samples, and $d$ is the number of features.

- **Coefficient of Determination ($R^2$)**:  Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables. It ranges from 0 to 1, with values closer to 1 indicating a better fit.
  $R^2 = r^2 = \frac{\text{SSR}}{\text{SST}} \in [0,1]$

  As $r^2$ approaches 1, the model's fit to the data improves.

### Ridge and Lasso Regression & Regularization

Ridge and Lasso Regression are extensions of linear regression designed to address **overfitting** and improve model generalization, especially when dealing with complex datasets or multicollinearity.  They achieve this by adding a **regularization term** to the Loss function, which penalizes large weights.

Overfitting occurs when a model learns the training data too well, including its noise, leading to poor performance on unseen data. Regularization helps to constrain the model, making it less sensitive to noise and thus more likely to generalize well.

**Why Regularization?**

1. **Preventing Overfitting:** By penalizing large weights, regularization discourages overly complex models that might fit the training data noise.
2. **Improving Model Stability:** Regularization can help with numerical stability, particularly when dealing with matrices that are nearly singular (non-invertible) due to multicollinearity or small eigenvalues. In linear regression, finding the optimal weights involves inverting the matrix $(X^T X)$. If this matrix is close to singular (determinant close to zero), the inverse can be unstable and lead to very large weights. Regularization helps to stabilize this inversion process.

**Regularization Penalty:**

Both Ridge and Lasso Regression add a penalty term to the standard Mean Squared Error (MSE) loss function:

$L_y(w) = ||\tilde{X} \cdot \hat{\omega} - y||_2^2 + \lambda_2 \cdot ||\tilde{\omega}||_2^2 + \lambda_1 \cdot ||\tilde{\omega}||_1$

Here, the Loss function consists of two parts:

1. **Data Fidelity Term:** $ ||\tilde{X} \cdot \hat{\omega} - y||_2^2 $ -  Measures how well the model fits the training data (Sum of Squared Errors).
2. **Regularization Term:** $ \lambda_2 \cdot ||\tilde{\omega}||_2^2 + \lambda_1 \cdot ||\tilde{\omega}||_1 $ - Penalizes the magnitude of the weights $\tilde{\omega}$.

- $\lambda_1$ and $\lambda_2$ are **regularization parameters** that control the strength of the penalty. Higher values increase the penalty, leading to smaller weights.
- $||\tilde{\omega}||_1$ is the L1 norm of the weight vector (sum of absolute values of weights). Using the L1 norm leads to **Lasso Regression**.
- $||\tilde{\omega}||_2^2$ is the squared L2 norm of the weight vector (sum of squared values of weights). Using the L2 norm leads to **Ridge Regression**.

**Types of Regularization:**

- **Ridge Regression (L2 Regularization):** Adds an L2 penalty term ($\lambda_2 \cdot ||\tilde{\omega}||_2^2$). Ridge regression shrinks all weights towards zero, but typically does not force them to be exactly zero. It is effective in reducing the impact of multicollinearity and reducing model complexity.

- **Lasso Regression (L1 Regularization):** Adds an L1 penalty term ($\lambda_1 \cdot ||\tilde{\omega}||_1$). Lasso regression can drive some weights to exactly zero, effectively performing feature selection by excluding less relevant features from the model.

- **Elastic Net Regularization:** Combines both L1 and L2 penalties. It balances the feature selection properties of Lasso and the weight shrinkage of Ridge, and can be particularly useful when dealing with datasets that have many correlated features.

**Key Differences between Ridge and Lasso:**

| Feature          | Ridge Regression (L2)                  | Lasso Regression (L1)                    |
|-------------------|------------------------------------------|-------------------------------------------|
| Penalty Term     | $\lambda_2 \cdot ||\tilde{\omega}||_2^2$ | $\lambda_1 \cdot ||\tilde{\omega}||_1$ |
| Weight Shrinkage | Shrinks weights towards zero              | Shrinks weights, can force some to zero   |
| Feature Selection| No automatic feature selection          | Performs feature selection                |
| Multicollinearity| Effective in reducing impact             | Less effective in multicollinearity       |

**Mathematical Solution for Ridge Regression:**

The weight vector $\hat{\omega}$ that minimizes the Ridge regression loss function has a closed-form solution:

$\hat{\omega} = (X^T X + \lambda_2 I)^{-1} X^T \cdot y$

where $I$ is the identity matrix. The addition of $\lambda_2 I$ to $X^T X$ ensures that the matrix $(X^T X + \lambda_2 I)$ is invertible, even if $X^T X$ is not, thus addressing numerical instability issues.

For any given $\lambda$, there exists a threshold $T$ such that minimizing $L_y(w) = ||\tilde{X} \cdot \hat{\omega} - y||_2^2 + \lambda||\tilde{\omega}||_k^k$ is equivalent to minimizing the standard least squares loss $L_y(2) = ||\tilde{X} \cdot \hat{\omega} - y||_2^2$ under the constraint $||\hat{\omega}||_2^2 \leq T$.  (Note: $\hat{\omega}$ includes $w_0$, while $\tilde{\omega}$ typically excludes $w_0$ in regularization terms).

### Loss Functions

The loss function measures the model's prediction error. Common loss functions include:

1) **Sum of Squared Errors (SSE)**:
   $\sum(t_i - h(x_i))^2$

2) **Mean Squared Error (MSE)**:
   $\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i-\hat{y}_i)^2$

3) **Root Mean Squared Error (RMSE)**:
   $\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i-\hat{y}_i)^2}$

4) **Mean Absolute Error (MAE):**
   $\text{MAE} = \frac{1}{n}\sum |y - \hat{y}|$

5) **R-squared:** Proportion of variance in the target explained by the model.

### Gradient Descent

Gradient Descent is an optimization algorithm used to minimize the loss function by iteratively adjusting model parameters.

#### Steps:
1. Initialize the parameters (w⃗, w₀) randomly.
2. Calculate the gradient of the loss function with respect to the parameters.
3. Update the parameters ($w_i$) in the opposite direction of the gradient to minimize the loss.

The update rule is:
$\Delta w_i = -\lambda \cdot \frac{\partial \text{Loss}(w)}{\partial w_i}$

Where:
- $\lambda$ is the learning rate
- $\frac{\partial \text{Loss}(w)}{\partial w_i}$ is the partial derivative of the loss function with respect to $w_i$.

GD is an off-line algorithm, meaning we process all data before making an update. This makes it computationally expensive with large datasets.

### Stochastic Gradient Descent

In contrast to regular Gradient Descent, Stochastic Gradient Descent (SGD) processes data in mini-batches, making it more efficient for large datasets.

#### Steps:
1. Initialize parameters (w⃗, w₀).
2. Shuffle and divide the dataset into m mini-batches.
3. For each mini-batch, calculate h(x) and update the parameters.
4. Repeat for multiple epochs (passes through the data).

The update rule for SGD is:
$\Delta w_i = -\frac{\lambda}{m} \cdot \sum(t_i - y_i)x_i$

Where:
- m is the mini-batch size

SGD is more efficient than GD for large datasets because it updates parameters more frequently, potentially leading to faster convergence.

The gradient computation for cross entropy loss in SGD:
$\Delta w_i = -\frac{\lambda}{m} \cdot \sum\left(\frac{\partial \text{Loss}(t_i,y_i)}{\partial w_i}\right)$

### Feature Normalization

Feature normalization is critical for gradient descent to work efficiently, especially when features have different scales. It helps the algorithm converge faster and prevents features with larger scales from dominating.

#### Methods:
1) **Min-Max Scaling**:
   $x_{\text{normalized}} = \frac{x - \min(x)}{\max(x) - \min(x)}$

2) **Mean Normalization**:
   $x_{\text{normalized}} = x - \text{mean}(x)$

3) **Standard Deviation Scaling**:
   $x_{\text{normalized}} = \frac{x}{\text{SD}(x)}$

4) **Standard Normalization (Z-score)**:
   $x_{\text{normalized}} = \frac{x - \text{mean}(x)}{\text{SD}(x)}$

   This centers the data around 0 with a standard deviation of 1.

## Classification

### Classification Overview

The goal of classification is to learn a function h(x⃗) that minimizes the loss function for dataset D = {(x_i, t_i)}, where t_i represents the class label and y = P(t=1|x) is the probability of class 1 given input x.

The key difference between classification and regression is that classification predicts categorical outputs rather than continuous values.

## Logistic Regression

### Logistic Regression Overview

Logistic Regression is used for binary classification. It models the probability of a positive class by applying the sigmoid function to a linear combination of features:

$$p(x) = \frac{1}{1 + e^{-(w_0 + w_1 x_1 + \dots + w_p x_p)}}$$

A probability above a chosen threshold (typically 0.5) indicates the positive class.

From a computational learning perspective, unlike linear regression, in logistic regression, we're not looking for a straight line to separate the data, but rather to categorize the data into categories (typically binary, receiving 0 or 1).

### Mathematical Formulation (Logistic Regression)

Here, the prediction uses the **sigmoid function** ($sigm: (-\infty, \infty) \rightarrow [0,1]$):
$\hat{y} = \frac{1}{1+e^{-(\hat{\omega} \cdot \tilde{X})}} = sigmoid(\hat{\omega} \cdot \tilde{X})$

The sigmoid function is defined as:
$y = g(z) = \frac{1}{1+e^{-z}}$

where $z = w₀ + \sum w_i x_i$

The sigmoid function $g(z)$ maps any real input to a value between 0 and 1, representing the probability of the positive class.

![Logistic Regression Curve](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/Exam_pass_logistic_curve.svg/1280px-Exam_pass_logistic_curve.svg.png)
*Figure: The sigmoid function mapping inputs to probabilities.*

### Key Concepts (Logistic Regression)

- **Sigmoid Function:** Transforms any real value into a probability between 0 and 1.
- **Binary Classification:** Predicts one of two possible classes.
- **Odds Ratio:** Exponentiating coefficients provides the factor change in odds for a unit change in a feature.
- **Maximum Likelihood Estimation:** The training method used to optimize model parameters.

**Advantage**: Provides a probability estimate for classification, not just a class label.
**Disadvantage**: May struggle with highly non-linear decision boundaries as it is fundamentally a linear classifier.

The decision boundary in logistic regression is linear and occurs where $w⃗ \cdot x⃗ = 0$. The distance from a point $x⃗$ to the decision boundary can be expressed as:
$\frac{w⃗ \cdot x⃗}{||w⃗||} = \frac{1}{||w⃗||} \cdot d(x⃗, w⃗)$

Where $d(x⃗, w⃗)$ represents the distance from point $x⃗$ to the decision boundary.

[![An Quick Intro to Logistic Regression](https://img.youtube.com/vi/EKm0spFxFG4/maxresdefault.jpg)](https://youtu.be/EKm0spFxFG4)

### Loss Function (Logistic Regression)

The **Loss function** for logistic regression, also known as **Binary Cross-Entropy**, is used to quantify the error between predicted probabilities and true labels:
$L(w) = -\sum \log(\hat{y}_i^{y_i}(1 - \hat{y}_i)^{1-y_i}) = ... = (-\sum y_i \log(sigm(w \cdot x_i)) + (1-y_i) \log((1-sigm(w \cdot x_i))))$

### Gradient Descent (Logistic Regression)

To find the weight vector $w$ that minimizes the Loss function, we use the **Gradient Descent** optimization algorithm. This iterative method starts with an initial guess for $w_0$ and refines it over iterations to reach the optimal $w$:
$w_i = w_{i-1} - \alpha\nabla_w L(w_{i-1})$

Here, $\alpha$ is the learning rate, and $\nabla_w L(w_{i-1})$ is the gradient of the Loss function with respect to $w$ at the previous iteration $w_{i-1}$.

### Cross Entropy

**Cross Entropy (CE)** is the standard loss function for logistic regression, measuring the dissimilarity between the predicted probability distribution and the actual distribution of classes.

For a single data point $(x,t)$, the Cross Entropy loss $C(y,t)$ is given by:

$C(y,t) = -t \cdot \log(y) - (1-t) \cdot \log(1-y) =
\begin{cases}
-\log(y) & \text{if } t=1 \\
-\log(1-y) & \text{if } t=0
\end{cases}$

Where:
- $y$ is the predicted probability from the logistic regression model.
- $t$ is the true label (0 or 1).

The total Cross Entropy loss for a dataset $D$ with $m$ examples is the average loss over all examples:
$\text{CE}_D(w) = \frac{1}{m} \cdot \sum C(y_i, t_i)$

**Perplexity** is a measure of how well a probability distribution predicts a sample. In the context of Cross Entropy, perplexity can be defined as:
$\text{Perplexity} = e^{\text{CE}_D(w)}$

### Gradient Descent with Cross Entropy

To minimize the Cross Entropy loss, we use Gradient Descent to iteratively update the weights. The gradient of the Cross Entropy loss function with respect to each weight $w_i$ is:

$\frac{\partial \text{CE}_D(t,y)}{\partial w_i} = \frac{\partial \text{CE}_D(t,y)}{\partial y} \cdot \frac{\partial y}{\partial w_i} = \frac{1}{m} \cdot \sum(y_i - t_i)x_i$

The weight update rule in Gradient Descent becomes:
$\Delta w_i = -\frac{\lambda}{m} \cdot \sum(t_i - y_i)x_i$

#### Comparison of GD and SGD with Cross Entropy:

When using Cross Entropy loss, the weight update rule remains similar for both Gradient Descent (GD) and Stochastic Gradient Descent (SGD), but the scope of the summation differs:

- **GD (Batch Gradient Descent)**: Computes the gradient over the entire dataset ($D$).
  $\Delta w_i = -\frac{\lambda}{m} \cdot \sum_{(x_i, t_i) \in D} (y_i - t_i)x_i$, where $m = |D|$ (size of the full dataset).

  **Advantages of GD**: Provides a more stable estimate of the gradient in each iteration as it uses the entire dataset. This can lead to more consistent convergence behavior.
  **Disadvantages of GD**: Can be computationally expensive and slow for large datasets because it requires processing all data points in each iteration.

- **SGD (Stochastic Gradient Descent)**: Computes the gradient and updates weights for each mini-batch (or even each data point in its purest form).
  $\Delta w_i = -\frac{\lambda}{m} \cdot \sum_{(x_i, t_i) \in \text{minibatch}} (y_i - t_i)x_i$, where $m = |\text{minibatch}|$ (size of the mini-batch, typically much smaller than the full dataset).

  **Advantages of SGD**: Much faster iterations compared to GD, especially on large datasets, as it processes only a small subset of data in each update. Can escape local minima more readily due to the noisy gradient updates.
  **Disadvantages of SGD**: The noisy updates can lead to oscillations in the loss function and convergence path, and may require careful tuning of the learning rate and mini-batch size.

### Multi-Category Classification

Extending binary classification to problems with more than two categories requires different approaches. Here are common strategies for multi-category classification:

#### One vs. Rest (One vs. All)

In the **One vs. Rest** (OvR) or **One vs. All** (OvA) strategy, we decompose the multi-class problem into multiple binary classification problems. For each class, we train a separate binary classifier to distinguish that class from all other classes combined.

- **Training Phase**: For $n$ classes, train $n$ binary classifiers.
  - For each class $i$, create a binary dataset where:
    - Examples of class $i$ are labeled as positive (1).
    - Examples from all other classes are labeled as negative (0).
  - Train a binary classifier (e.g., Logistic Regression) on each of these binary datasets.

- **Prediction Phase**: To classify a new input $x$:
  - Run all $n$ binary classifiers on $x$.
  - Choose the class corresponding to the classifier that outputs the highest probability or confidence score.

**Advantage**: Simplicity and ease of implementation. It allows the use of any binary classifier for multi-class problems. Extends naturally from binary classifiers.
**Disadvantage**: Can suffer from class imbalance, especially if one class is much smaller than the others, as negative examples will often outnumber positive ones significantly in each binary problem. Also, confidence scores from different classifiers might not be directly comparable.

### One vs. One Multi-Class Classification

The **One vs. One** (OvO) strategy is another approach to reduce a multi-class classification problem into multiple binary classification problems. In OvO, we train a binary classifier for every pair of classes.

- **Training Phase**: For $n$ classes, train $n(n-1)/2$ binary classifiers.
  - For each pair of classes $(i, j)$, where $i \neq j$:
    - Create a binary dataset consisting only of examples from classes $i$ and $j$.
    - Label examples from class $i$ as positive and examples from class $j$ as negative (or vice versa).
    - Train a binary classifier on this dataset to distinguish between classes $i$ and $j$.

- **Prediction Phase**: To classify a new input $x$:
  - Run all $n(n-1)/2$ binary classifiers on $x$.
  - Aggregate the predictions from all classifiers to make a final classification decision.

There are several methods to combine the results from these pairwise classifiers:

#### 1. **Majority Vote**

In **Majority Vote**, each classifier trained on classes $(i, j)$ effectively "votes" for either class $i$ or class $j$. For a new input $x$, we count the number of votes each class receives across all pairwise classifiers. The class that receives the most votes is chosen as the final prediction.

#### 2. **Probabilistic Combination**

For **Probabilistic Combination**, if the binary classifiers can output probability estimates, we can combine these probabilities to get a more refined prediction. For each class $i$, we consider all the classifiers that were trained to distinguish class $i$ from another class. We can then average or sum the probabilities of class $i$ being chosen over all these pairwise classifiers to get an overall probability estimate for class $i$. The class with the highest combined probability is selected.

#### 3. **Decision-Based Weighting**

**Decision-Based Weighting** involves assigning weights to each pairwise classifier's decision based on its confidence or performance. For instance, classifiers that are more accurate on a validation set or have higher confidence scores can be given more weight in the final decision. This allows more reliable classifiers to have a greater influence on the outcome.

### Multiclass Classification Extensions

For problems inherently multiclass, there are extensions and methods designed to handle multiple categories directly, rather than decomposing into binary problems.

#### MCCE (Multiclass Cross-Entropy)

**Multiclass Cross-Entropy (MCCE)** is a generalization of binary cross-entropy for multiclass classification problems. It directly measures the dissimilarity between the predicted probability distribution over all classes and the true class distribution. MCCE is used as a loss function in models designed for multiclass output, such as neural networks with a softmax output layer.

#### Softmax + CCE (Categorical Cross-Entropy)

The **Softmax function**, when combined with **Categorical Cross-Entropy (CCE)** loss, is a prevalent approach for multiclass classification, particularly in neural networks.

- **Softmax Function**: Takes raw scores (logits) from the model's output layer and converts them into a probability distribution over all classes. For an input $x$, if the model outputs logits $z = [z_1, z_2, ..., z_n]$ for $n$ classes, the softmax function computes probabilities $\hat{y} = [\hat{y}_1, \hat{y}_2, ..., \hat{y}_n]$ as:
  $\hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$
  This ensures that each $\hat{y}_i$ is in the range $[0, 1]$ and $\sum_{i=1}^{n} \hat{y}_i = 1$, forming a valid probability distribution.

- **Categorical Cross-Entropy (CCE) Loss**: Measures the loss between the predicted probability distribution $\hat{y}$ and the true class distribution $t$. If $t$ is a one-hot encoded vector (e.g., for class $k$, $t = [0, 0, ..., 1, ..., 0]$ with 1 at the $k^{th}$ position), the CCE loss for a single example is:
  $C(\hat{y}, t) = -\sum_{i=1}^{n} t_i \log(\hat{y}_i) = -\log(\hat{y}_k)$ if $t$ is one-hot encoded for class $k$.
  The total CCE loss over a dataset is the average of the losses for all examples.

#### Unified Classifier CE

A **Unified Classifier** approach involves training a single model that is inherently designed to output probabilities for all classes directly. For example, a neural network with a softmax output layer trained with Categorical Cross-Entropy is a unified classifier. Unlike OvR and OvO, which train multiple binary classifiers, a unified classifier learns to discriminate between all classes simultaneously in a single training process. This approach is often more efficient and can capture inter-class relationships more effectively than decomposed methods.

## Model Evaluation

### Bias vs. Variance Trade-off

The **Bias-Variance Trade-off** is a central concept in machine learning that highlights a critical balance in model building. It describes the relationship between two types of errors that models can make: errors from overly simplistic assumptions (**bias**) and errors from excessive sensitivity to training data fluctuations (**variance**). Achieving a good model involves minimizing the total error by optimally balancing bias and variance.

Mathematically, the **Expected Error** of a model can be decomposed into three components:

$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$

- **Bias**: Bias refers to the error introduced by approximating a real-world problem, which may be complex, by a simplified model. High bias models make strong assumptions about the data and are typically simpler, with less flexibility to fit complex relationships. A high bias model tends to **underfit** the data, meaning it fails to capture the underlying patterns in the training data and performs poorly on both training and test sets.

  **Underfitting** occurs when the model is too simple compared to the complexity of the data. In such cases, the model has high bias and typically low variance.

- **Variance**: Variance refers to the error due to the model's sensitivity to small fluctuations in the training dataset. High variance models are complex and flexible, capable of fitting the training data very closely, including the noise present in it. However, this sensitivity makes them perform poorly on unseen data, as they have essentially memorized the noise from the training set. High variance models tend to **overfit** the training data.

  **Overfitting** happens when the model is excessively complex and fits the training data too well, including its noise. Overfitted models have low bias but high variance.

- **Irreducible Noise**: This component of error is inherent in the problem itself and cannot be reduced by any model, regardless of its complexity. It accounts for factors that are beyond the model's control, such as data collection errors or randomness in the underlying process.

![Bias–Variance Trade-off](https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Bias_and_variance_contributing_to_total_error.svg/1024px-Bias_and_variance_contributing_to_total_error.svg.png)
*Figure: Illustration of bias and variance contributions to total error. Models with different complexity show varying levels of bias and variance, affecting the total error.*

### Key Performance Indicators (KPIs)

**Key Performance Indicators (KPIs)**, also known as **performance metrics**, are crucial for evaluating the effectiveness and efficiency of machine learning models. They provide quantifiable measures of how well a model is performing on a given task, whether it's classification or regression.

#### Classification Metrics:

**Confusion Matrix:** A table that summarizes the performance of a classification model by showing the counts of true positive (TP), true negative (TN), false positive (FP), and false negative (FN) predictions.

|                      | Predicted Positive | Predicted Negative |
|----------------------|--------------------|--------------------|
| **Actual Positive**  | True Positive (TP) | False Negative (FN)|
| **Actual Negative**  | False Positive (FP)| True Negative (TN) |

Based on the confusion matrix, several KPIs can be derived:

- **Accuracy:** The ratio of correctly classified instances to the total number of instances.
  $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$

- **Precision:** Measures the proportion of predicted positives that are actually positive. It is a measure of result exactness.
  $\text{Precision} = \frac{TP}{TP+FP}$

- **Recall (Sensitivity or True Positive Rate):** Measures the proportion of actual positives that are correctly identified. It is a measure of completeness.
  $\text{Recall} = \frac{TP}{TP+FN}$

- **F1 Score:** The harmonic mean of precision and recall, providing a balanced measure of a test's accuracy. It is especially useful when dealing with imbalanced datasets.
  $F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$

- **Specificity (True Negative Rate):** Measures the proportion of actual negatives that are correctly identified.
  $\text{Specificity} = \frac{TN}{TN+FP}$

- **ROC AUC (Area Under the ROC Curve):** The Area Under the Receiver Operating Characteristic (ROC) curve. ROC is a probability curve that plots the TPR against the FPR at various threshold values and AUC represents the degree or measure of separability. It summarizes the trade-off between the true positive rate and the false positive rate for different thresholds. A higher AUC indicates better classifier performance, with AUC=1 representing a perfect classifier and AUC=0.5 representing performance no better than random guessing.

#### Regression Metrics:

For regression models, KPIs assess the accuracy of predicted numerical values compared to actual values:

- **Mean Squared Error (MSE):** The average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value.
  $\text{MSE} = \frac{1}{n}\sum (y - \hat{y})^2$

- **Root Mean Squared Error (RMSE):** The square root of the MSE. RMSE is more interpretable than MSE because it is in the original units of the output variable.
  $\text{RMSE} = \sqrt{\text{MSE}}$

- **Mean Absolute Error (MAE):** The average of the absolute differences between predicted and actual values. MAE is less sensitive to outliers than MSE because it does not square the errors.
  $\text{MAE} = \frac{1}{n}\sum |y - \hat{y}|$

- **R-squared (Coefficient of Determination):** Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. It ranges from 0 to 1, with higher values indicating a better fit of the model to the data.
  $R^2$

### Feature Selection

When facing issues of **high bias** or **underfitting**, it's often beneficial to revisit the features used in the model. **Feature selection** and **feature engineering** are crucial steps to improve model performance.

#### 1. **Increasing Feature Complexity**

If the model is underfitting, it might be too simplistic to capture the underlying patterns in the data. Increasing the complexity of features can help. This involves:

- **Adding Polynomial Features**: Including higher-degree polynomial terms of existing features (e.g., $x, x^2, x^3$). This can help capture non-linear relationships.
- **Using More Complex Transformations**: Applying non-linear transformations to features, such as logarithmic, exponential, or trigonometric functions, to better align with the data's underlying structure.

#### 2. **Adding Interaction Features**

Sometimes, the relationships between features and the target variable are not just additive but also interactive. **Interaction features** are created by combining two or more features to capture these interactions. For example, if you have features 'age' and 'income', an interaction feature could be 'age * income', which might be relevant if the effect of age on the target variable depends on the income level.

When dealing with a large number of features, it's important to assess their relevance and importance. **Feature ranking** helps in identifying which features contribute most significantly to the model's predictive power. Features that lead to a greater reduction in error are considered more important. Feature importance can be evaluated through:

- **Statistical Methods**: Using statistical tests to assess the correlation or mutual information between each feature and the target variable.
- **Model-Based Importance**: Many machine learning models (like decision trees and ensemble methods) provide built-in measures of feature importance, based on how much each feature contributes to reducing the error during training.
- **Performance Impact Measurement**: Systematically evaluating the impact of each feature on the model's performance. This could involve training the model multiple times, each time with a different subset of features, and observing how the performance changes. Features that, when removed, cause a significant drop in performance are considered important.

### Validation & Cross-Validation

**Validation** is a critical step in machine learning to ensure that a model not only performs well on the training data but also generalizes to unseen data. Proper validation helps in selecting the best model, tuning hyperparameters, and estimating the model's performance in real-world scenarios.

The dataset is typically partitioned into three sets:

- **Training Set**: Used to train the machine learning model. The model learns the parameters from this data.
- **Validation Set**: Used to tune hyperparameters and compare different models. It helps in making decisions about model complexity and preventing overfitting to the training set. The validation set is used to evaluate the model's performance during training and model selection, but it is not used for training itself.
- **Test Set**: A completely held-out dataset used only for the final evaluation of the model's performance after model selection and hyperparameter tuning are complete. It provides an unbiased estimate of the model's generalization performance on unseen data.

**Hold-Out Validation**: The simplest form of validation is the **hold-out method**, where the dataset is split into a training set and a test set (or sometimes training, validation, and test sets). The model is trained on the training set and evaluated on the test set. While straightforward, it has limitations:

- **Single Split Dependency**: Performance metrics are based on a single train-test split, which may not be representative of the overall dataset, especially if the dataset is small or if the data split is not random enough.
- **Variance in Performance Estimate**: The performance estimate can vary significantly depending on how the data is split.

**k-Fold Cross-Validation**: To overcome the limitations of hold-out validation, **k-Fold Cross-Validation** provides a more robust and reliable estimate of model performance. In k-fold cross-validation, the dataset is divided into $k$ equally sized folds. The model is trained and evaluated $k$ times. In each iteration (fold):

1. One fold is used as the **validation set** (or test set in some contexts).
2. The remaining $k-1$ folds are combined to form the **training set**.
3. The model is trained on the training set and evaluated on the validation set.
4. The performance metric is recorded for each fold.

After $k$ iterations, we will have $k$ performance metrics. The final performance estimate is typically the average of these $k$ values.

![K-Fold Cross-Validation](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/K-fold_cross_validation_EN.svg/1024px-K-fold_cross_validation_EN.svg.png)
*Figure: Illustration of k-fold cross-validation where the dataset is partitioned into k subsets. In each fold, one subset is used for validation, and the rest are used for training.*

#### Key Concepts:

- **Training Set:** The portion of data used to train the model, i.e., to learn the model parameters.
- **Validation Set:** Used for tuning hyperparameters and selecting the best model among different configurations. It helps prevent overfitting to the training data.
- **Test Set:** A completely independent dataset used for the final, unbiased evaluation of the trained model's performance.
- **Hold-Out Validation:** A simple validation technique where data is split into training and test sets once.
- **k-Fold Cross-Validation:** A more robust validation technique that divides the data into $k$ folds and performs $k$ iterations of training and validation.
- **Hyperparameter Tuning:** The process of selecting the optimal values for hyperparameters (model settings that are not learned from data, e.g., learning rate, regularization strength) based on validation performance. Cross-validation is often used to evaluate different hyperparameter settings.

[![Cross Validation](https://img.youtube.com/vi/fSytzGwwBVw/maxresdefault.jpg)](https://youtu.be/fSytzGwwBVw)

### Data Preprocessing, Feature Engineering & Encoding

Effective **data preprocessing**, **feature engineering**, and **feature encoding** are essential steps in preparing data for machine learning models. These steps ensure that the data is in a suitable format, improve model performance, and can significantly impact the success of a machine learning project.

#### Data Cleansing:

**Data cleansing**, or data cleaning, is the process of identifying and correcting or removing errors, inconsistencies, and inaccuracies in a dataset. It is a foundational step in data preprocessing and is crucial for building reliable and accurate models. Common data cleansing tasks include:

- **Handling Missing Values**:
  - **Imputation**: Filling in missing values with statistical measures (mean, median, mode), predicted values from a model, or using algorithms designed for missing data imputation.
  - **Removal**: Removing rows or columns with missing values. This should be done cautiously as it can lead to loss of valuable data, especially if missing values are not completely at random.
- **Outlier Detection and Treatment**:
  - **Detection**: Identifying data points that significantly deviate from the norm using statistical methods (e.g., Z-score, IQR) or visualization techniques (e.g., box plots, scatter plots).
  - **Treatment**: Deciding how to handle outliers, which could involve removal, transformation (e.g., log transformation to reduce the impact of extreme values), or capping (setting extreme values to a certain threshold).
- **Format Standardization**: Ensuring data consistency by standardizing formats for dates, times, currencies, addresses, and other data types. This includes correcting inconsistencies in spelling, capitalization, and units of measurement.

#### Feature Creation (Engineering):

**Feature engineering** is the process of transforming raw data into features that are more suitable and effective for machine learning models. It requires domain knowledge and creativity and is often where significant improvements in model performance can be achieved. Feature engineering techniques include:

- **Polynomial Features**: Creating new features by raising existing features to a power or including interaction terms between features. For example, converting a linear feature into quadratic or cubic features to capture non-linear relationships.
- **Date Feature Extraction**: Extracting meaningful components from date and time variables, such as year, month, day of the week, hour, minute, and whether it's a holiday or weekend.
- **Aggregation Features**: Creating summary statistics from related features or grouping variables. For example, calculating the average, sum, maximum, minimum, or standard deviation of transaction amounts for each customer.
- **Domain-Specific Features**: Creating features based on domain knowledge that are relevant to the problem. For instance, in natural language processing, features like TF-IDF scores, word embeddings, or sentiment scores.

*Impact:* Well-engineered features can significantly enhance a model's ability to learn complex patterns from data, leading to improved accuracy, generalization, and interpretability. Conversely, poorly chosen or engineered features can lead to underfitting or overfitting, and obscure the true relationships in the data.

#### One-Hot Encoding:

**One-Hot Encoding** is a crucial technique for converting categorical variables into a numerical format that machine learning algorithms can process. Many algorithms work best, or exclusively, with numerical inputs. One-hot encoding transforms each category of a categorical variable into a new binary variable (feature).

- **Process**:
  1. Identify the unique categories in a categorical variable.
  2. For each unique category, create a new binary column.
  3. For each observation, place a '1' in the column corresponding to its category and '0' in all other category columns.

*Example*: Consider a 'Color' feature with categories {'Red', 'Green', 'Blue'}. One-hot encoding would transform this into three binary features: 'Is_Red', 'Is_Green', 'Is_Blue'.

| Original Color | Is_Red | Is_Green | Is_Blue |
|----------------|--------|----------|---------|
| Red            | 1      | 0        | 0       |
| Green          | 0      | 1        | 0       |
| Blue           | 0      | 0        | 1       |
| Red            | 1      | 0        | 0       |

*Importance*: One-hot encoding is essential for categorical variables because it:
- Converts categories into a format that numerical algorithms can use.
- Prevents algorithms from assuming ordinal relationships between categories (which might be implied if categories were simply mapped to integers).
- Increases the dimensionality of the dataset but often improves model performance by properly representing categorical information.

#### Train–Validation–Test Split:

**Train-Validation-Test Split** is a fundamental practice in machine learning to ensure models are evaluated robustly and fairly. The process of splitting the dataset into training, validation, and test sets serves distinct purposes in the model development lifecycle:

- **Training Set**: The largest portion of the dataset, used to train the machine learning model. The model learns the parameters (weights and biases) from this data by optimizing the loss function.

- **Validation Set**: A separate portion of the dataset used to tune hyperparameters and perform model selection. Hyperparameters are settings of the model that are not learned from the data but are set prior to training (e.g., learning rate, regularization parameters, complexity parameters in trees). The validation set helps to evaluate how well the model generalizes to unseen data during training and allows for adjustments to hyperparameters to improve performance. By evaluating on the validation set, we can avoid overfitting to the training set and choose a model configuration that performs best on data it has not been directly trained on.

- **Test Set**: A completely independent, held-out portion of the dataset that is used only for the final evaluation of the trained model. The test set provides an unbiased estimate of the model's generalization performance on unseen, real-world data. It should be used only once, at the very end of the model development process, after all model selection and hyperparameter tuning have been completed using the training and validation sets. This ensures that the performance metric on the test set truly reflects how well the model is expected to perform in practice.

**Cross-Validation for Refined Performance Estimates**: In scenarios where data is limited, or for more reliable performance estimation, **cross-validation** techniques (like k-fold cross-validation) can be used, especially in conjunction with the training and validation sets. Cross-validation provides a more robust estimate of model performance by training and validating the model multiple times across different subsets of the data.

## Fighting High Variance (Overfitting)

**High variance**, or **overfitting**, is a common problem in machine learning where a model learns the training data too well, including its noise, and consequently performs poorly on unseen data. Strategies to combat high variance and improve model generalization include:

- **Regularization**: Regularization techniques add a penalty term to the loss function to discourage overly complex models. Common regularization methods include:
  - **L1 Regularization (Lasso)**: Adds a penalty proportional to the absolute value of the coefficients. It can lead to sparse models with feature selection.
  - **L2 Regularization (Ridge)**: Adds a penalty proportional to the square of the coefficients. It shrinks the coefficients towards zero but does not typically set them exactly to zero.
  - **Elastic Net Regularization**: A hybrid approach that combines L1 and L2 regularization to balance feature selection and coefficient shrinkage.

- **Ensemble Methods**: Ensemble methods combine predictions from multiple models to reduce variance and improve overall performance. Common ensemble techniques include:
  - **Bagging (Bootstrap Aggregating)**: Training multiple instances of the same model on bootstrapped subsets of the training data and averaging their predictions. Random Forests are a popular example of bagging.
  - **Boosting**: Sequentially training models where each subsequent model tries to correct the errors of its predecessors. Examples include AdaBoost, Gradient Boosting Machines (GBM), and XGBoost.

- **Early Stopping**: In iterative training algorithms (like gradient descent for neural networks), **early stopping** involves monitoring the model's performance on a validation set during training. Training is halted when the validation performance starts to degrade (e.g., validation loss increases), even if the training loss is still decreasing. This prevents the model from continuing to learn noise in the training data, which leads to overfitting.

- **Feature Selection/Dimensionality Reduction**: Reducing the number of input features can simplify the model and prevent overfitting, especially when dealing with high-dimensional data. Feature selection techniques aim to choose the most relevant features, while dimensionality reduction methods (like Principal Component Analysis - PCA) transform the feature space into a lower-dimensional space while preserving most of the variance.

- **Increasing Training Data**: Providing more training data to the model can help it learn the underlying patterns better and reduce overfitting. With more data, the model is less likely to memorize noise and more likely to generalize to unseen examples. However, collecting more data may not always be feasible or cost-effective.

[![L1 Vs L2 Regularzation Methods](https://img.youtube.com/vi/aBgMRXSqd04/maxresdefault.jpg)](https://youtu.be/aBgMRXSqd04)

## Unsupervised Classification Methods

### K-Nearest Neighbors (KNN)

### KNN Overview

**K-Nearest Neighbors (KNN)** is a type of instance-based learning, or lazy learning, algorithm. Unlike eager learning algorithms that build a model explicitly during the training phase, KNN defers the learning process until a query is made. It's used for both classification and regression tasks. KNN predicts the label or value of a new data point based on the labels or values of its *k* nearest neighbors in the training dataset, using a distance metric.

From a computational learning perspective, K-Nearest Neighbors (KNN) is used to classify new data points by identifying their similarity to existing points in the dataset. It's particularly useful for datasets where decision boundaries are irregular or non-linear.

**KNN Illustration:** [PLACEHOLDER FOR KNN ILLUSTRATION: Shows a point with a circle around it containing k nearest neighbors]

Imagine a new data point you want to classify. KNN works by:

1. **Finding Neighbors**: Identifying the $k$ closest data points from the training set to the new point, based on a chosen distance metric. These closest points are the 'neighbors'. The value of $k$ is a user-defined constant.
2. **Classification**: Determining the class of the new point based on the classes of its $k$ neighbors. In classification, this is typically done by majority voting—the class that is most frequent among the $k$ neighbors is assigned to the new point.

The choice of $k$ is critical.
- A **small $k$ value** (like $k=1$) makes the model highly sensitive to noise and outliers in the training data, leading to complex decision boundaries and potentially overfitting.
- A **large $k$ value** smooths out the decision boundaries, reducing sensitivity to noise but potentially leading to underfitting, especially if the decision boundaries are complex.

#### Key Points (KNN):

- **Distance Metrics:** The choice of distance metric is crucial as it determines how 'near' neighbors are defined. Common distance metrics include:
  - **Euclidean Distance**: The straight-line distance between two points in Euclidean space. Most commonly used.
  - **Manhattan Distance**: The sum of the absolute differences of their Cartesian coordinates. Useful when movement is restricted to grid-like paths.
  - **Cosine Similarity**: Measures the cosine of the angle between two vectors, useful for high-dimensional data like text documents, where magnitude is less important than direction.

- **Parameter $k$:** The number of neighbors to consider. A crucial hyperparameter that affects the bias-variance trade-off. Optimal $k$ is usually chosen through cross-validation.

- **Normalization:** Feature scaling (normalization or standardization) is essential for KNN, especially when features are measured in different units or have vastly different ranges. Without normalization, features with larger scales can disproportionately influence the distance calculations.

- **Curse of Dimensionality:** In high-dimensional spaces, the concept of 'nearest neighbors' can become less meaningful because distances between points become more uniform (data becomes sparse). This is known as the curse of dimensionality and can degrade the performance of KNN.

[![K nearest Neighbors](https://img.youtube.com/vi/b6uHw7QW_n4/maxresdefault.jpg)](https://youtu.be/b6uHw7QW_n4)

#### KNN Algorithm

The K-Nearest Neighbors algorithm for classification works as follows:

1. **Distance Calculation**:
   - Given a new, unclassified data point $x$, calculate the distance between $x$ and every data point $x_i$ in the training dataset. Use a chosen distance metric, such as Euclidean distance:
     $d(x, x_i) = \sqrt{\sum_{j=1}^{p} (x_j - x_{ij})^2}$
     where $p$ is the number of features, $x_j$ is the $j^{th}$ feature of the new point $x$, and $x_{ij}$ is the $j^{th}$ feature of the $i^{th}$ training point $x_i$.

2. **Neighbor Selection**:
   - Select the $k$ training data points that are nearest to $x$, i.e., those with the smallest distances $d(x, x_i)$.

3. **Prediction**:
   - **For Classification**: Determine the class label based on the majority class among the $k$ nearest neighbors. Count the number of neighbors belonging to each class. Assign the class that appears most frequently among the $k$ neighbors as the class label for the new point $x$. If there is a tie in class counts, tie-breaking mechanisms (like choosing the class of the closest neighbor or random selection) can be used.
   - **For Regression**: For regression problems, the KNN algorithm predicts the value for the new data point by calculating the average (or median) of the target values of its $k$ nearest neighbors.

### Bayesian Classifiers

**Bayesian classifiers** are a family of probabilistic classifiers that apply **Bayes' Theorem** to predict the probability of each class for a given data point. They are based on the principle that a classifier can predict class membership probabilities, which can be more informative than just assigning a class label. The most well-known Bayesian classifier is the **Naive Bayes classifier**.

**Bayes' Theorem** provides a way to update beliefs (probabilities) given evidence. In the context of classification, it allows us to calculate the probability of a class $C$ given a set of features $X$, based on prior probabilities and likelihoods:

$P(C \mid X) = \frac{P(X \mid C) P(C)}{P(X)}$

Where:
- $P(C \mid X)$ is the **posterior probability** of class $C$ given features $X$. This is what we want to calculate—the probability that a data point with features $X$ belongs to class $C$.
- $P(X \mid C)$ is the **likelihood** of observing features $X$ given that the class is $C$. It measures how likely the features are to occur if the data point belongs to class $C$.
- $P(C)$ is the **prior probability** of class $C$. This is the probability of class $C$ before observing any features, often estimated from the frequency of classes in the training data.
- $P(X)$ is the **marginal likelihood** or evidence, the probability of observing features $X$. In classification, $P(X)$ serves as a normalization factor to ensure that the posterior probabilities sum to one across all classes, and often does not need to be calculated explicitly for decision making (as it is the same for all classes for a given $X$).

#### Key Points:

- **Naive Bayes Assumption**: The "naive" aspect of Naive Bayes classifiers comes from the assumption of **conditional independence** between features given the class. It assumes that the presence or absence of a particular feature is unrelated to the presence or absence of any other feature, given the class label. Mathematically, this assumption simplifies the calculation of the likelihood $P(X \mid C)$ as:
  $P(X \mid C) = P(x_1, x_2, ..., x_n \mid C) = \prod_{i=1}^n P(x_i \mid C)$
  where $X = (x_1, x_2, ..., x_n)$ is the feature vector. This assumption is often not true in real-world data, but Naive Bayes classifiers can still perform surprisingly well in many applications, especially text classification.

- **Naive Bayes Classifiers are Fast and Effective**: Despite the oversimplified assumption, Naive Bayes classifiers are computationally efficient, easy to implement, and can perform well, particularly in text classification and for high-dimensional data. They require less training data compared to more complex models.

- **Laplace Smoothing**: To handle the issue of zero probabilities (e.g., when a feature value is not seen in the training data for a particular class), **Laplace smoothing** (or add-one smoothing) is commonly used. It adds a small count (typically 1) to each feature count for each class, ensuring that no probability is exactly zero. This prevents the entire product in the likelihood calculation from becoming zero if any feature probability is zero.

- **Text Classification Applications**: Naive Bayes is particularly popular and effective in **text classification** tasks, such as spam filtering, sentiment analysis, and document categorization. In text classification, features are often word counts or TF-IDF scores. A common approach is to use the "bag-of-words" model, where the order of words is ignored, and the presence and frequency of each word are used as features.

### Decision Trees & Random Forests

**Decision Trees** and **Random Forests** are powerful and versatile machine learning methods used for both classification and regression tasks. They are particularly valued for their interpretability and ability to handle complex datasets with non-linear relationships.

**Decision Trees**: A decision tree is a flowchart-like structure where each internal node represents a test on an attribute (feature), each branch represents the outcome of the test, and each leaf node represents a class label (for classification) or a predicted value (for regression). The paths from root to leaf represent classification rules.

#### Key Points (Decision Trees):

- **Splitting Data**: Decision trees work by recursively partitioning the data into subsets based on the values of input features. The goal is to create partitions such that data points within each partition are as homogeneous as possible with respect to the target variable.

- **Split Criteria**: At each node, the algorithm chooses the best feature to split the data and the threshold for the split. The "best" split is determined by criteria that measure the impurity of the data at each node. Common impurity measures include:
  - **Information Gain (for Classification)**: Used in algorithms like ID3 and C4.5. It measures the reduction in entropy after splitting the data based on a feature. Entropy measures the disorder or impurity in a set of examples.
  - **Gini Impurity (for Classification)**: Used in algorithms like CART (Classification and Regression Trees). Gini impurity measures the probability of misclassifying a randomly chosen element in the dataset if it were randomly labeled according to the class distribution in the subset.
  - **Variance Reduction (for Regression)**: In regression trees, the criterion is often to choose splits that minimize the variance of the target variable in the resulting subsets.

- **Interpretability**: One of the main advantages of decision trees is their interpretability. The tree structure makes it easy to understand the decision-making process and the rules used by the model. You can trace the path from the root to a leaf to see which feature tests lead to a particular prediction.

![Decision Tree Example](https://github.com/user-attachments/assets/ffada2e4-eeac-4c18-945d-49abb7118930)

*Figure: A decision tree example for the Iris dataset. This tree classifies iris flowers into different species based on petal and sepal measurements.*

- **Pruning**: Decision trees are prone to overfitting, especially if they are allowed to grow very deep. **Pruning** is a technique to reduce the size of decision trees and prevent overfitting. Pruning involves removing branches of the tree that do not improve generalization performance, typically by evaluating performance on a validation set.

[![Bagging vs Boosting](https://github.com/user-attachments/assets/c600802c-82b6-4b7a-818f-e74e1e427be1)](https://youtu.be/tjy0yL1rRRU)

**Random Forests**: Random Forests are an ensemble learning method that builds multiple decision trees and merges their predictions to get a more accurate and stable prediction. Random Forests are designed to reduce variance and prevent overfitting that is common with single decision trees.

#### Key Points (Random Forests):

- **Bagging (Bootstrap Aggregating)**: Random Forests use a technique called **bagging**. Bagging involves creating multiple subsets of the training data by random sampling with replacement (bootstrap sampling). For each subset, a decision tree is trained independently.

- **Random Feature Selection**: In addition to bagging, Random Forests introduce randomness in feature selection. When building each tree, at each node, instead of considering all possible features for splitting, a random subset of features is selected. The best feature to split the node is chosen from this random subset. This further decorrelates the trees in the forest and reduces variance.

- **Aggregation of Predictions**: After training multiple decision trees, Random Forests aggregate their predictions to make a final prediction.
  - **For Classification**: Random Forests typically use **majority voting**. Each tree in the forest predicts a class label, and the class with the most votes becomes the final prediction.
  - **For Regression**: Random Forests average the predictions of all trees in the forest to get the final predicted value.

- **Feature Importance**: Random Forests provide a measure of **feature importance**. By tracking how much each feature reduces impurity across all trees in the forest, Random Forests can estimate the importance of each feature in the prediction process. This can be useful for feature selection and understanding which features are most influential in the model.

![Random Forest Illustration](https://github.com/user-attachments/assets/032b91d7-7bb0-4978-8973-e6edbe75c89c)

*Figure: Multiple decision trees in a Random Forest. Each tree is trained on a random subset of data and features, and their predictions are aggregated for the final output.*

### Support Vector Machines (SVM)

### SVM Overview

**Support Vector Machines (SVM)** are a powerful and versatile class of supervised learning algorithms used for classification, regression, and outlier detection. SVMs are particularly effective in high-dimensional spaces and are known for their robustness and efficiency. The fundamental concept behind SVM is to find an optimal hyperplane that separates different classes in the feature space with the maximum possible margin.

From a computational learning perspective, Support Vector Machines (SVM) are used to find a linear boundary that optimally separates different classes of data. The key idea is to maximize the margin around this boundary, which improves generalization.

We aim to find an optimal separating line of the form $x \cdot \hat{a} - d = 0$, where $\hat{a} = \frac{(a_1,a_2)}{\sqrt{a_1^2+a_2^2}}$ is a unit normal vector to the line. This line should maximize the margin, which is the distance from the line to the nearest data points of each class. The constraints are given by $y_i \cdot (x_i\hat{a} - d) \geq 1$, ensuring that all data points are on the correct side of the margin. The distance between the supporting lines $x \cdot \hat{a} - d = \pm 1$ is $\frac{-2}{||w||_2}$, or simply 2 if $||w||_2 = 1$. To maximize the margin, we need to minimize $||w||_2$.

When data is not perfectly linearly separable, SVMs can use a **soft margin** approach. This approach tolerates some misclassifications (data points within the margin or on the wrong side of the boundary) by introducing slack variables and a penalty for these violations, balancing the goal of maximizing the margin with minimizing classification errors.

For datasets where linear separation is not possible, SVMs utilize the **Kernel trick**. Kernel SVMs map the original input data into a higher-dimensional feature space using a kernel function. In this higher-dimensional space, it's often possible to find a linear hyperplane that separates the classes. The beauty of the kernel trick is that this mapping is done implicitly; we don't need to compute the coordinates of the data points in the higher-dimensional space explicitly.

#### Kernels (SVM)

**Kernels** are functions that define the similarity or inner product between data points in the transformed feature space. By choosing different kernel functions, SVMs can model complex, non-linear decision boundaries. Common kernel functions include:

1. **Linear Kernel**: The simplest kernel, equivalent to a linear SVM without any transformation.
   $K(u, v) = \langle u, v \rangle = u^T \cdot v$
   Suitable for linearly separable data and often used as a baseline.

2. **Polynomial Kernel**: Maps data to a higher-dimensional space using polynomial combinations of the original features.
   $K(u, v) = (1 + \langle u, v \rangle)^d$, where $d > 0$ is the degree of the polynomial.
   Effective for problems where data can be separated by polynomial curves or surfaces.

3. **Gaussian Kernel (Radial Basis Function - RBF Kernel)**: Maps data to an infinite-dimensional space. The Gaussian kernel is defined as:
   $K(u, v) = e^{-\frac{||u-v||^2}{2\sigma^2}}$
   where $||u-v||^2$ is the squared Euclidean distance between $u$ and $v$, and $\sigma$ is a free parameter that controls the kernel's width. The Gaussian kernel is very flexible and can model complex decision boundaries. It's a popular choice for non-linear SVMs.

4. **Bag of Words (BOW) Kernel**: Used in text classification. BOW represents documents as vectors of word counts. A kernel can be defined to measure the similarity between document vectors.

   For a dictionary of words $D = \{w_i: i = 1, ..., 100,000\}$, the BOW representation of a document $S = (w_1, ..., w_n)$ is a feature vector $\Phi(S)$ where each element counts the occurrences of a word from $D$ in $S$:
   $\Phi(w_1, ..., w_n) = \Phi(S) = (\#\{w_i \in S \mid w_i = w_j\} \text{ for } j = 1, ..., 100,000)$

   The similarity between two documents $u$ and $v$ can be measured using a kernel based on their BOW representations, such as the cosine similarity kernel:
   $K(u, v) = \frac{\langle\Phi(u),\Phi(v)\rangle}{||\Phi(u)||_2 ||\Phi(v)||_2}$
   This kernel measures the cosine of the angle between the BOW vectors, indicating the similarity in word frequency profiles, irrespective of document length.

5. **TF-IDF Kernel**: Term Frequency-Inverse Document Frequency (TF-IDF) is another weighting scheme used in text processing to reflect the importance of a term in a document relative to a collection of documents (corpus). TF-IDF addresses some limitations of the basic Bag of Words approach, such as giving too much weight to common words.

   TF-IDF assigns a weight to term $w_i$ in document $d$ as:
   $t_i = TF(w_i, d) \cdot IDF(w_i)$

   - **Term Frequency (TF)**: Measures how frequently a term occurs in a document. A common formula is:
     $TF(w_i, d) = \log(1 + \text{number of occurrences of } w_i \text{ in document } d)$

   - **Inverse Document Frequency (IDF)**: Measures how important a term is across a corpus of documents. It downweights terms that are very common across documents and upweights terms that are rare.
     $IDF(w_i) = \log\left(\frac{|D|}{\sum_{j} I(x_{ij})}\right)$
     where $|D|$ is the total number of documents in the corpus, and $\sum_{j} I(x_{ij})$ is the number of documents in which term $w_i$ appears (document frequency). $I(x_{ij})$ is an indicator function that is 1 if term $w_i$ appears in document $j$, and 0 otherwise.

   Using TF-IDF vectors $\Phi_{TF-IDF}(u)$ and $\Phi_{TF-IDF}(v)$ for documents $u$ and $v$, a TF-IDF kernel can be defined, often using cosine similarity to measure document similarity based on TF-IDF weighted term vectors.

## Unsupervised Learning

### Unsupervised Learning Overview

**Unsupervised learning** deals with datasets that are not labeled, meaning there are no target outputs provided. The goal in unsupervised learning is to discover hidden patterns, structures, or relationships within the data itself. Unlike supervised learning, which aims to predict or classify outputs based on inputs, unsupervised learning seeks to understand the inherent organization of the data.

### Principal Component Analysis (PCA)

### PCA Overview

**Principal Component Analysis (PCA)** is a powerful dimensionality reduction technique widely used in unsupervised learning. PCA aims to reduce the number of dimensions in high-dimensional data while preserving as much of the variance (information) as possible. It achieves this by identifying a new set of variables, called **principal components**, which are linear combinations of the original variables and are orthogonal to each other. The principal components are ordered such that the first few components capture most of the variance in the original data.

From a computational learning perspective, Principal Component Analysis (PCA) reduces the dimensionality of a dataset by finding a new set of orthonormal basis vectors (principal components) that capture the maximum variance in the data. This simplifies the problem while retaining essential information.

In PCA, we aim to find orthonormal eigenvectors $v_1, ..., v_n$ (where $\langle v_i, v_j \rangle = 1$ if $i=j$ and 0 if $i \neq j$) that correspond to the maximal eigenvalues of the matrix $\tilde{X}^T\tilde{X}$, where $\tilde{X} = X - \mu$ (mean-centered data matrix). The magnitude of the eigenvalue associated with each eigenvector indicates the amount of variance explained by that principal component. Components are ordered by eigenvalue magnitude, with the first component explaining the most variance, the second component explaining the second most variance, and so on.

The transformation $T$ from the original data space to the reduced space is a linear transformation given by:
$T_{v_1,...,v_n}(x) = \hat{x} = \sum_{i=1}^{l} \langle x, v_i \rangle v_i = \sum_{i=1}^{l} v_i^T \cdot x \cdot v_i = \sum_{i=1}^{l} v_i \cdot v_i^T \cdot x = (\sum_{i=1}^{l} v_i \cdot v_i^T) \cdot x$
where $l$ is the number of principal components chosen for dimensionality reduction, and $v_1, ..., v_l$ are the first $l$ principal components (eigenvectors corresponding to the top $l$ eigenvalues).

The goal is to minimize the **reconstruction error**, which is the difference between the original data points and their reconstructions from the reduced-dimensional representation:
Reconstruction error: $\sum_{i=1}^{n} ||x_i - \hat{x}_i||_2^2$

### Mathematical Formulation (PCA)
#### Process:

1. **Standardize the data**: Ensure each feature has zero mean and unit variance. This step is crucial because PCA is sensitive to the scale of the features.
2. **Compute the covariance matrix**: Calculate the covariance matrix of the standardized dataset. The covariance matrix summarizes the relationships between different features.
3. **Perform eigenvalue decomposition**: Compute the eigenvectors and eigenvalues of the covariance matrix. Eigenvectors represent the principal components (directions of maximum variance), and eigenvalues represent the amount of variance explained by each principal component.
4. **Select principal components**: Sort the eigenvectors by their eigenvalues in descending order. Choose the top $k$ eigenvectors that correspond to the largest eigenvalues. These top eigenvectors are the principal components that will be used to reduce dimensionality.
5. **Project the data**: Project the original data onto the subspace spanned by the selected principal components. This is done by multiplying the standardized data matrix by the matrix of top $k$ eigenvectors. The result is a lower-dimensional representation of the data that retains most of the original variance.

#### Usage:

- **Dimensionality Reduction**: Reduce the number of features to simplify models, speed up computation, and reduce storage space.
- **Data Visualization**: Reduce data to 2 or 3 dimensions for visualization, allowing for exploration of data patterns and structures in scatter plots.
- **Noise Reduction**: PCA can help in filtering out noise by focusing on the principal components that capture the most variance, which is often assumed to be signal rather than noise.
- **Feature Extraction for Modeling**: Use principal components as new features for other machine learning models, such as classifiers or regression models. This can improve model performance by mitigating multicollinearity and focusing on the most informative aspects of the data.
- **Mitigating Multicollinearity**: In datasets with highly correlated features, PCA can transform them into a set of uncorrelated principal components, addressing multicollinearity issues that can affect the stability and interpretability of models like linear regression.

### K-Means Clustering

**K-Means Clustering** is a popular unsupervised learning algorithm used to partition a dataset into $K$ distinct, non-overlapping clusters. The goal of K-Means is to minimize the within-cluster variance (or sum of squared distances), making clusters as compact and separate as possible.

#### Key Concepts:

- **Centroids**: Each cluster is represented by its centroid, which is the mean of all data points in the cluster. Centroids act as the center of gravity for their respective clusters.
- **Assignment**: Data points are assigned to the cluster whose centroid is nearest to them, typically using Euclidean distance.
- **Iteration**: K-Means is an iterative algorithm. It starts with an initial set of centroids and iteratively refines them in two main steps: assignment and update, until convergence.

#### Algorithm Steps for K-Means Clustering:

1. **Initialization**: Randomly select $K$ initial centroids. Common initialization methods include:
   - **Random Selection**: Randomly choose $K$ data points from the dataset to serve as initial centroids.
   - **K-Means++**: A more sophisticated initialization method that spreads out the initial centroids to improve convergence speed and clustering quality.

2. **Assignment Step**: Assign each data point to the cluster with the nearest centroid. For each data point $x_i$, calculate the distance to each centroid $\mu_j$ (e.g., Euclidean distance $||x_i - \mu_j||^2$) and assign $x_i$ to the cluster $C_j$ with the minimum distance.

3. **Update Step**: Recalculate the centroids for each cluster by taking the mean of all data points assigned to that cluster. For each cluster $C_j$, update its centroid $\mu_j$ to be the mean of all points in $C_j$:
   $\mu_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i$

4. **Iteration**: Repeat steps 2 and 3 until convergence. Convergence is typically determined by one of the following conditions:
   - **Centroids no longer change significantly**: The centroids from one iteration to the next are practically the same.
   - **Data point assignments do not change**: Data points remain in the same clusters from one iteration to the next.
   - **Maximum number of iterations reached**: A predefined number of iterations is completed.

#### Key Considerations for K-Means:

- **Choosing K**: Determining the optimal number of clusters $K$ is a critical challenge in K-Means, as it is not automatically determined by the algorithm. Techniques to help choose $K$ include:
  - **Elbow Method**: Plot the within-cluster sum of squares (WCSS) for different values of $K$. The "elbow" point in the plot, where the rate of decrease in WCSS sharply changes, is often suggested as a reasonable choice for $K$.
  - **Silhouette Score**: Measures how similar each point in one cluster is to points in its assigned cluster, compared to points in other clusters. The silhouette score ranges from -1 to +1, with higher values indicating better-defined clusters.

- **Limitations**:
  - **Assumes Spherical Clusters**: K-Means assumes that clusters are spherical, equally sized, and have similar densities. It may not perform well on clusters that are elongated, irregularly shaped, or have varying densities.
  - **Sensitive to Initialization**: The initial placement of centroids can significantly affect the final clustering outcome. K-Means may converge to a local minimum rather than the global minimum, especially if initial centroids are poorly chosen. Running K-Means multiple times with different initializations and selecting the best result (e.g., based on WCSS) is a common practice.
  - **Requires Numerical Data**: K-Means is typically applied to numerical data and requires features to be in a comparable scale. Categorical data needs to be appropriately encoded into numerical form before using K-Means.

### Hierarchical Clustering

**Hierarchical Clustering** is another unsupervised learning algorithm used to group data points into clusters. Unlike K-Means, which partitions data into a pre-specified number of clusters, hierarchical clustering builds a hierarchy of clusters, represented as a tree-like structure called a **dendrogram**. Hierarchical clustering does not require specifying the number of clusters beforehand and can reveal hierarchical relationships between data points.

#### Key Approaches:

Hierarchical clustering methods are broadly divided into two types:

- **Agglomerative (Bottom-Up) Approach**:
  1. **Initialization**: Start with each data point as a separate cluster. Initially, if there are $N$ data points, there are $N$ clusters.
  2. **Merge**: Iteratively merge the two closest clusters into a single cluster. The definition of "closest" is based on a chosen **linkage criterion**.
  3. **Repeat**: Repeat step 2 until all data points are in a single cluster, or until a desired number of clusters is reached.

- **Divisive (Top-Down) Approach**:
  1. **Initialization**: Start with all data points in a single cluster.
  2. **Split**: Iteratively split the most heterogeneous cluster into two. The "most heterogeneous" cluster and the way to split it are defined by certain criteria.
  3. **Repeat**: Repeat step 2 until each cluster contains only one data point, or until a desired number of clusters is reached.

#### Linkage Criteria:

In agglomerative hierarchical clustering, the **linkage criterion** defines how the distance between clusters is calculated. Different linkage criteria can lead to different cluster structures:

- **Single Linkage (Minimum Linkage)**: The distance between two clusters is defined as the minimum distance between any point in one cluster and any point in the other cluster. Single linkage can lead to long, straggly clusters and is sensitive to noise.

- **Complete Linkage (Maximum Linkage)**: The distance between two clusters is defined as the maximum distance between any point in one cluster and any point in the other cluster. Complete linkage tends to produce compact clusters and is less sensitive to noise than single linkage.

- **Average Linkage**: The distance between two clusters is the average of the distances between all pairs of points, where each pair consists of one point from each cluster. Average linkage is a compromise between single and complete linkage.

- **Ward's Method**: Ward's linkage criterion aims to minimize the increase in total within-cluster variance after merging two clusters. It tends to produce compact and spherical clusters and is often preferred when clusters are expected to be relatively uniform in size.

## Linear Algebra Fundamentals

From a computational learning perspective, **Linear Algebra** is not just a prerequisite but a foundational language for expressing and understanding many machine learning algorithms. Concepts from linear algebra are deeply woven into the fabric of machine learning, providing the mathematical framework for data representation, algorithm design, and optimization.

### Inner/Scalar Product

In $ \mathbb{R}^d $ (d-dimensional real space), the **inner product** or **scalar product** of two vectors $u$ and $v$, denoted as $u \cdot v$ or $\langle u, v \rangle$, is a fundamental operation defined as:

$u \cdot v \triangleq \langle u, v \rangle = \sum_{i} u_i \cdot v_i = u_0 \cdot v_0 + u_1 \cdot v_1 + ... + u_d \cdot v_d$

where $u = (u_0, u_1, ..., u_d)$ and $v = (v_0, v_1, ..., v_d)$.

Additionally, the inner product has a geometric interpretation related to the angle between the vectors:
$\langle u, v \rangle = ||u|| \cdot ||v|| \cdot \cos(\theta)$
where $||u||$ and $||v||$ are the magnitudes (L2 norms) of vectors $u$ and $v$ respectively, and $\theta$ is the angle between them.

### L-norms

**L-norms** are functions that measure the "length" or "magnitude" of vectors in a vector space. Two commonly used L-norms in machine learning are the L1 norm and the L2 norm:

- **L2 Norm (Euclidean Norm)**: The L2 norm of a vector $u$, denoted as $||u||_2$, is defined as:
  $||u||_2 \triangleq \sqrt{\langle u, u \rangle} = \sqrt{u_0^2 + u_1^2 + ... + u_n^2}$
  It corresponds to the standard Euclidean length of the vector.

- **L1 Norm (Manhattan Norm or Taxicab Norm)**: The L1 norm of a vector $u$, denoted as $||u||_1$, is defined as:
  $||u||_1 \triangleq \sum_{i} |u_i| = |u_0| + |u_1| + ... + |u_n|$
  It is the sum of the absolute values of the vector components.

Note: When we write $||u||_2^2$, we mean the square of the L2 norm, which is simply $||u||_2^2 = \langle u, u \rangle = \sum_{i} u_i^2$.

### Metric Distance

A **metric distance** is a function that defines a concept of distance between elements of a set. For vectors $u$ and $v$ in a vector space, a common metric distance is derived from a norm, such as the L2 norm:

Metric distance: $d(u, v) = ||u - v||$

Using the L2 norm, the Euclidean distance between vectors $u$ and $v$ is:
$d(u, v) = ||u - v||_2 = \sqrt{\sum_{i} (u_i - v_i)^2}$

Linearity of Operations:

Many operations in linear algebra, including the inner product and norms, exhibit linearity properties. For example, the inner product is linear in both arguments:
$\langle a\bar{u} + b\bar{v}, \bar{w} \rangle = a \langle u, w \rangle + b \langle v, w \rangle$
$\langle \bar{w}, a\bar{u} + b\bar{v} \rangle = a \langle w, u \rangle + b \langle w, v \rangle$
where $a$ and $b$ are scalars, and $u, v, w$ are vectors. This linearity is fundamental to the efficiency and analytical tractability of many machine learning algorithms.

## Text Feature Extraction

### N-Gram Features

**N-gram features** are a fundamental concept in text processing and natural language processing (NLP). An N-gram is a contiguous sequence of $n$ items (words, characters, or tokens) from a given text or speech. N-grams are used to capture the local context of text and are valuable features for various text analysis tasks.

- **Usage**: N-grams are used extensively in text feature extraction to capture sequential patterns in text data. They help in:
  - **Text Classification**: Identifying topics, sentiment, or authorship.
  - **Language Modeling**: Predicting the next word in a sequence.
  - **Information Retrieval**: Improving search relevance by matching sequences of words.
  - **Spell Checking and Autocorrect**: Detecting and correcting errors based on common word sequences.

- **Types of N-grams**:
  - **Unigrams (1-gram)**: Sequences of single words. Example: "word".
  - **Bigrams (2-gram)**: Sequences of two consecutive words. Example: "machine learning".
  - **Trigrams (3-gram)**: Sequences of three consecutive words. Example: "natural language processing".
  - **Character N-grams**: Sequences of characters, used for language identification, spell correction, and handling out-of-vocabulary words.

- **Trade-Off**:
  - **Context vs. Dimensionality**: Higher order n-grams (larger $n$) capture more context and sequential information in the text, which can be beneficial for tasks requiring understanding of word order and phrases. However, they also lead to a significant increase in the dimensionality of the feature space. The number of possible n-grams can grow exponentially with $n$ and the vocabulary size, leading to sparse feature vectors and increased computational cost.
  - **Sparsity**: Feature vectors based on higher order n-grams are often sparser, as many possible n-grams may not appear frequently in the corpus.

### TF-IDF (Term Frequency–Inverse Document Frequency)

**TF-IDF (Term Frequency–Inverse Document Frequency)** is a numerical statistic that is used to reflect how important a word is to a document in a collection or corpus. TF-IDF is widely used in information retrieval, text mining, and user modeling. It weights terms based on their frequency in a document and their inverse document frequency across the corpus, emphasizing terms that are frequent in a specific document but rare across the entire corpus.

- **TF (Term Frequency)**: Measures how often a term appears in a document. It is calculated as the number of times a term $t$ appears in a document $d$. A common approach to normalize term frequency is to use the logarithmically scaled TF:
  $TF(t, d) = \log(1 + \text{number of occurrences of term } t \text{ in document } d)$
  The logarithm scaling helps to dampen the effect of very frequent terms within a document.

- **IDF (Inverse Document Frequency)**: Measures how important a term is across a collection of documents (corpus). It reduces the weight of terms that are very common in the corpus and increases the weight of terms that are rare. IDF for a term $t$ is calculated as:
  $IDF(t) = \log\left(\frac{|D|}{\sum_{j} I(d_j \text{ contains } t)}\right)$
  where $|D|$ is the total number of documents in the corpus, and $\sum_{j} I(d_j \text{ contains } t)$ is the number of documents in the corpus that contain the term $t$ (document frequency). $I(d_j \text{ contains } t)$ is an indicator function that is 1 if document $d_j$ contains term $t$, and 0 otherwise. The logarithm is used to dampen the effect of IDF.

- **TF-IDF Calculation**: The TF-IDF weight for a term $t$ in a document $d$ is then calculated by multiplying its TF and IDF values:
  $TF-IDF(t, d) = TF(t, d) \times IDF(t)$

- **Application**: TF-IDF is used to:
  - **Enhance Text Classification**: By weighting words, TF-IDF highlights terms that are distinctive to particular documents within a corpus, improving the performance of classifiers.
  - **Information Retrieval**: In search engines, TF-IDF helps to rank documents according to their relevance to a query. Documents that contain query terms with high TF-IDF weights are considered more relevant.
  - **Keyword Extraction**: Identifying the most important words in a document by selecting terms with high TF-IDF scores.

---

# Further Reading

## Sklearn Documentation

![image](https://github.com/user-attachments/assets/a41a4bff-325d-44e6-9dd9-6cc862d94057)

[Sklearn Documentation link - Press here](https://scikit-learn.org/stable/supervised_learning.html)

#
[Deep Learning Study Guidebook](https://github.com/ShovalBenjer/Housing_Price_Prediction_Advanced_Regresson_Kaggle)

## Books
[The Elements of Statistical Learning, Data mining, inference and predicition - Trevor Hastie](https://github.com/ShovalBenjer/Machine_Learning_Study_Guidebook)

[An introduction to Statistical learning with applications in Python - Gareth James, Daniella Witten](https://github.com/ShovalBenjer/Machine_Learning_Study_Guidebook/blob/main/docs/An%20Introduction%20to%20Statistical%20Learning_Gareth%20James.pdf)

# Practical Tools & Resources

### Kaggle
A platform for data science competitions, datasets, and collaborative learning.
- **Competitions:** Real-world problems with leaderboards.
- **Datasets:** Diverse and extensive collections for practice.
- **Notebooks & Forums:** Community sharing of code and strategies.

Check my past kaggle projects:

[Regresson Methods](https://github.com/ShovalBenjer/Housing_Price_Prediction_Advanced_Regresson_Kaggle)

[Classification Methods](https://github.com/ShovalBenjer/Titanic---Machine-Learning-from-Disaster)

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
