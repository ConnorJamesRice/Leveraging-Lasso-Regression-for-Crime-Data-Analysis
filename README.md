# Leveraging-Lasso-Regression-for-Crime-Data-Analysis

## Introduction

This GitHub README is a comprehensive exploration of the application of Lasso regression to a real-world dataset encompassing crime statistics in 1,994 US communities. This dataset is a treasure trove of information, offering insights into the complex interplay of socio-economic and demographic factors that shape violent crime rates. Our journey embarks on a mission to not only understand the nuances of this dataset but to showcase the potential of Lasso regression in addressing significant challenges in the field of data analysis.

## Getting Started

Before diving into the analysis, it's crucial to understand the fundamental underpinnings of the Lasso regression and the iterative shrinkage thresholding algorithm that we will implement to solve it. The Lasso problem, in this context, is defined as follows:

Given λ > 0 and data (xi, yi) for i = 1 to n, the Lasso problem is to find:

arg min w∈Rd, b∈R Σi=1 to n ((xiᵀw + b - yi)² + λ * Σj=1 to d |wj|)

In our programming tasks, we will implement the Iterative Shrinkage Thresholding Algorithm (ISTA) to solve the Lasso problem. This algorithm is a variant of the subgradient descent method and is described by Algorithm 1:

**Algorithm 1: Iterative Shrinkage Thresholding Algorithm for Lasso**

Input: Step size η

```
while not converged do
  b0 ← b - 2η Σi=1 to n (xiᵀw + b - yi)
  
  for k ∈ {1, 2, · · · d} do
    w0k ← wk - 2η Σi=1 to n xi,k(xiᵀw + b - yi)
    
    w0k ←
    
    
    
    
    w0k + 2ηλ  if w0k < -2ηλ
    0 if -2ηλ ≤ w0k ≤ 2ηλ
    w0k - 2ηλ  if w0k > 2ηλ
    end
  b ← b0, w ← w0
end
```

Before we dive into the practical implementation, here are some useful hints to consider:

- Utilize matrix libraries for matrix operations instead of loops, especially when updating w. The use of numpy functions can significantly improve efficiency.
- Regularly check if the objective value is non-increasing at each step.
- Define a suitable stopping condition. Typically, the algorithm stops when no element of w changes by more than a small δ during an iteration. Adjusting this condition can improve algorithm performance.
- Efficiently solve the Lasso problem on the same dataset for various values of λ to create a regularization path. To do this, start with a large λ and then decrease it by a constant ratio (e.g., a factor of 2) for each consecutive solution.
- Calculate the value of λmax using Equation (1) to determine the first λ in the regularization path.

## Investigating Synthetic Data

As a prelude to our analysis, we will begin by working with synthetic data that helps us understand the capabilities of the Lasso in distinguishing relevant and irrelevant features. The data generation process follows a model with the following specifications:

- n = 500 (number of data points)
- d = 1000 (number of features)
- k = 100 (number of relevant features)
- σ = 1 (noise level)

The data generation model follows the equation yi = wᵀxi + εᵢ, where w is defined as per Equation (2):

```
wj =
(
j/k if j ∈ {1, . . . , k}
0 otherwise
```

The εᵢ values are independently drawn from a normal distribution N(0, 1).

## Analyzing Lasso Solutions

### a. Analyzing Non-Zeros as a Function of λ

With our synthetic data, we embark on a journey to solve multiple Lasso problems on a regularization path. We start at λmax, where no features are selected (as per Equation (1)), and gradually decrease λ by a constant ratio (e.g., 2) until nearly all features are chosen. In Plot 1, we visualize the number of non-zeros as a function of λ on the x-axis, with a logarithmic scale for λ (Tip: use plt.xscale('log')).

### b. Evaluating False Discovery Rate (FDR) and True Positive Rate (TPR)

For each λ value we explore, we record the values for False Discovery Rate (FDR) and True Positive Rate (TPR). FDR is calculated as the number of incorrect non-zeros in wb divided by the total number of non-zeros in wb. TPR is computed as the number of correct non-zeros in wb relative to k (the number of relevant features). In Plot 2, we depict these values, with FDR on the x-axis and TPR on the y-axis.

### c. Observing the Impact of λ

The two plots in sections 'a' and 'b' provide a comprehensive view of how λ affects the Lasso solutions. In general, as we vary λ, we observe how the sparsity of the solution changes (plot 1) and how it impacts the trade-off between false discoveries and true positives (plot 2).

## Conclusion

In conclusion, the analysis of synthetic data provides us with a solid foundation to understand how Lasso regression can effectively distinguish between relevant and irrelevant features. The relationship between λ and the sparsity of the solution, as well as its impact on false discoveries and true positives, is crucial for making informed data-driven decisions.

**Historical Policies and Influential Features:**

Our analysis begins by examining features in the dataset that are influenced by historical policies and decisions. Among these, the "County" feature stands out as an emblem of local policy choices significantly impacting crime rates. The "PCTUnemployed" feature highlights the effect of policies aimed at reducing unemployment, as increased employment can deter criminal activities. Finally, "NumInShelters" reflects the efforts to shelter homeless individuals, which not only addresses a humanitarian issue but also plays a crucial role in reducing street crimes.

**Features as Causes or Consequences:**

An intricate aspect of this analysis involves distinguishing between features that may be interpreted as reasons for higher levels of violent crime and those that are more likely to be results or consequences of such crime. For example, the "State" feature captures the influence of the population within a state on crime rates. In contrast, "RacepctBlack" is deeply rooted in historical and societal factors. Similarly, "PCTHouOccup" is intertwined with demographics and the overall population in an area.

## Lasso Regression: A Powerful Tool

### A Glimpse into the Lasso Solver

Our journey into the world of Lasso regression unfolds with a careful examination of the Lasso solver. We employ a range of regularization parameters (λ) to understand how they impact the number of nonzero weights. This exploration is visualized through a plot, providing valuable insights into the sparsity of the model under varying levels of regularization.

### Unveiling the Regularization Paths

Continuing our analysis, we investigate the regularization paths of specific coefficients, including "agePct12t29," "pctWSocSec," "pctUrban," "agePct65up," and "householdsize." These plots offer a visual representation of how the magnitude of coefficients changes with varying levels of regularization, shedding light on the significance of these variables in predicting crime rates.

### Mean Squared Error Analysis

Our journey wouldn't be complete without a meticulous evaluation of the model's performance. We plot the Mean Squared Error (MSE) for both the training and test datasets across different values of λ. This analysis enables us to gauge the model's predictive accuracy under various levels of regularization.

## Interpreting Lasso Coefficients

In the process, we delve into the interpretation of Lasso coefficients. Intuitively, we anticipate that variables such as the percentage of kids born to never-married parents might have higher positive Lasso coefficients. Conversely, the percentage of kids in family households with two parents could exhibit lower coefficients. These trends reflect the impact of family structure on crime rates and offer valuable insights.

## Unmasking Flaws in Policy Recommendations

Finally, we address the implications of policy recommendations that emerge from our analysis. For instance, suggesting policies to relocate people aged 65 and up to high-crime areas as a crime reduction strategy might seem intuitive on the surface. However, a closer examination reveals a significant statistical flaw in this line of reasoning. Such a policy lacks empirical support and may, in fact, worsen the situation by potentially making these areas more attractive to criminals.

In conclusion, this GitHub README serves as a testament to the power of Lasso regression in deciphering intricate datasets and understanding the multifaceted factors that contribute to crime rates. Our analysis showcases the significance of thoughtful interpretation and the need for empirically grounded policy recommendations.
