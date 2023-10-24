# Leveraging Lasso Regression for Crime Data Analysis

## Introduction

This GitHub README is a comprehensive exploration of the application of Lasso regression to real-world crime data from 1,994 US communities. This dataset provides valuable insights into the intricate interplay of socio-economic and demographic factors influencing violent crime rates. Our journey delves into understanding this dataset while highlighting the potential of Lasso regression in tackling complex data analysis challenges.

## Getting Started

Before diving into our analysis, it's crucial to grasp the fundamentals of Lasso regression and the Iterative Shrinkage Thresholding Algorithm (ISTA), which we'll implement to solve the Lasso problem. The Lasso problem can be succinctly described as finding the optimal values for w and b by minimizing the following objective function:

![Lasso Objective Function](assets/lasso_objective.png)

In our implementation, we employ the ISTA to solve this problem iteratively. It's worth noting that we've provided hints for efficient coding, such as the use of matrix libraries for matrix operations and selecting a suitable stopping condition. Additionally, we'll explore a range of λ values to create a regularization path and find λmax to initiate this path.

## Investigating Synthetic Data

As a precursor to our analysis, we begin by working with synthetic data. This synthetic dataset helps us comprehend the Lasso's ability to distinguish between relevant and irrelevant features. Here are the key parameters for our synthetic data:

- n = 500 (data points)
- d = 1000 (features)
- k = 100 (relevant features)
- σ = 1 (noise level)

Our data generation model is driven by the equation yi = wᵀxi + εᵢ, where w is defined according to Equation (2):

![Definition of w](assets/w_definition.png)

The εᵢ values follow a normal distribution N(0, 1).

## Analyzing Lasso Solutions

### a. Analyzing Non-Zeros as a Function of λ

With our synthetic data, we embark on a journey to solve multiple Lasso problems across a regularization path. We initiate this path at λmax, where no features are selected, and gradually decrease λ. In Plot 1, we visually depict the number of non-zero coefficients as a function of λ. The logarithmic scale for λ (plt.xscale('log')) aids in comprehending the relationship.

### b. Evaluating False Discovery Rate (FDR) and True Positive Rate (TPR)

For each λ in our exploration, we meticulously record values for False Discovery Rate (FDR) and True Positive Rate (TPR). FDR measures the proportion of incorrect non-zeros in wb, relative to the total non-zeros. TPR quantifies the proportion of correct non-zeros in wb compared to k (the number of relevant features). Plot 2 offers a visual representation of these values, with FDR on the x-axis and TPR on the y-axis.

### c. Observing the Impact of λ

The two plots in sections 'a' and 'b' provide insights into how λ influences Lasso solutions. Variations in λ reveal shifts in the sparsity of the model (Plot 1) and impact the trade-off between false discoveries and true positives (Plot 2).

## Historical Policies and Influential Features

Our analysis commences by scrutinizing dataset features influenced by historical policies and decisions. Key highlights include:

- "County": Reflecting the influence of local policy choices on crime rates.
- "PCTUnemployed": Demonstrating the impact of policies aimed at reducing unemployment, potentially decreasing criminal activities.
- "NumInShelters": Highlighting the significance of sheltering homeless individuals, a move that addresses both humanitarian concerns and a reduction in street crimes.

## Features as Causes or Consequences

Distinguishing between features that could be interpreted as reasons for higher crime rates and those that result from these rates is a pivotal aspect of our analysis. For instance:

- "State": Captures the influence of a state's population on crime rates.
- "RacepctBlack": Reflects deep-rooted historical and societal factors.
- "PCTHouOccup": Tied to demographics and the local population.

## Lasso Regression: A Powerful Tool

### A Glimpse into the Lasso Solver

Our journey unfolds by meticulously examining the Lasso solver. We explore the impact of various regularization parameters (λ) on the number of non-zero coefficients. This exploration is visually presented, offering insights into the model's sparsity at different regularization levels.

### Unveiling the Regularization Paths

Continuing our analysis, we delve into the regularization paths for specific coefficients, such as "agePct12t29," "pctWSocSec," "pctUrban," "agePct65up," and "householdsize." These plots reveal how the magnitude of coefficients changes with varying regularization levels, shedding light on the significance of these variables in predicting crime rates.

### Mean Squared Error Analysis

Our analysis is incomplete without an in-depth evaluation of the model's performance. We present Mean Squared Error (MSE) plots for both training and test datasets, illustrating how predictive accuracy varies across different λ values.

### Interpreting Lasso Coefficients

In the process, we delve into the interpretation of Lasso coefficients. Our intuition suggests that variables like the percentage of kids born to never-married parents might have higher positive Lasso coefficients, reflecting their impact on crime rates. Conversely, the percentage of kids in family households with two parents could exhibit lower coefficients, indicating a potentially mitigating effect on crime rates.

### Unmasking Flaws in Policy Recommendations

In our analysis, we critically address the implications of policy recommendations that emerge. For example, proposing policies to relocate people aged 65 and up to high-crime areas as a crime reduction strategy may seem intuitive. However, a closer examination reveals a significant statistical flaw in this reasoning. Such a policy lacks empirical support and may potentially attract criminals to these areas.

## Conclusion

In conclusion, this GitHub README serves as a testament to the power of Lasso regression in deciphering intricate datasets. We've gained valuable insights into the multifaceted factors contributing to crime rates. Our analysis underscores the importance of thoughtful interpretation and emphasizes the need for empirically grounded policy recommendations.
