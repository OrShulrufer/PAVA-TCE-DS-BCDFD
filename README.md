
PAVA-TCE-DS-BCDFD: Predictive Modeling Toolkit
Welcome to the PAVA-TCE-DS-BCDFD repository! This toolkit introduces a comprehensive approach to probability calibration, leveraging the Pooled Adjacent Violators Algorithm (PAVA) combined with Total Calibration Error (TCE), Dynamic Significance (DS), and Binomial CDF Deviation (BCDFD) to ensure the reliability and accuracy of probabilistic models.

Overview
PAVA-TCE-DS-BCDFD is designed to enhance the reliability of predictive models by refining the predicted probabilities to align closely with actual outcomes. This toolkit is especially useful in fields such as finance, weather forecasting, and medical diagnostics, where precise probability estimates are crucial.

Features
Monotonicity: Ensures a logical progression of predicted probabilities.
Positive Percentile Cap: Maintains balance and avoids bias in skewed datasets.
Dynamic Significance Calculation: Adjusts significance levels based on data characteristics to enhance precision.
Normalization: Standardizes predictions to a common scale for uniform calibration.
Installation
Clone the repository and install the required dependencies:

bash
Copy code
git clone https://github.com/OrShulrufer/PAVA-TCE-DS-BCDFD.git
cd PAVA-TCE-DS-BCDFD
pip install -r requirements.txt
Usage
Here is a basic example of how to use the PAVA-TCE-DS-BCDFD toolkit:

python
Copy code
from pava_tce_ds_bcdfd import PAVA_TCE_DS_BCDFD_Calibrator

# Initialize the calibrator
calibrator = PAVA_TCE_DS_BCDFD_Calibrator(base_estimator=my_base_estimator, cv=5, n_bins=10)

# Fit the calibrator with predicted probabilities and actual outcomes
calibrator.fit(X_train, y_train)

# Calibrate new predictions
calibrated_probabilities = calibrator.predict_proba(X_test)
Step-by-Step Methodology for Calibration Using PAVA
1. Obtain Predicted Probabilities and Actual Outcomes
Collect the predicted probabilities from the classifier (e.g., logistic regression, random forest) and the actual outcomes.

2. PAVA Calibration
2.1 Monotonicity
A calibration curve is monotonic if the predicted probability of an event either consistently increases or remains the same as the underlying score or likelihood increases. This logical progression is essential for maintaining prediction integrity.

2.2 Pooled Adjacent Violators Algorithm (PAVA)
PAVA is a specific implementation of isotonic regression. It iteratively adjusts predicted probabilities by pooling adjacent bins that violate the monotonicity constraint, ensuring a smooth calibration curve.

2.2.1 Methodology
PAVA iteratively pools adjacent bins that do not satisfy the monotonicity condition, ensuring a monotonic relationship.

2.2.2 Advantages
Computationally efficient and easy to implement.
Ensures strict monotonicity in the calibration curve.
2.2.3 Disadvantages
Similar to isotonic regression, it can overfit sparse data, which needs to be addressed.
2.2.4 Real-life Example
Used in risk scoring systems in finance to ensure that higher scores correspond to higher risks.

3. Calibrator Components and Their Functions
3.1 Base Estimator
Provides the initial predictions that will be calibrated by the calibrator. Calibration refines these initial predictions to make them more accurate and reliable.

3.2 Cross-Validation
Splits the data into training and testing sets to validate the calibration process. This ensures the calibration parameters are robust and generalize well to new data, reducing the risk of overfitting.

3.3 Binning Optimization Components
3.3.1 PAVA-BC Algorithms (_pava_bc)
Ensures that the binning respects monotonicity and adjusts for binomial proportions, providing a solid foundation for further calibration.

3.3.1.1 Edge Bin Problem
Addresses the issue where the first and last bins tend to have higher deviations due to fewer data points.

3.3.1.2 Last Bin Last Huge Range Problem
Handles the problem where the last bin often covers a large range, potentially skewing the calibration. By implementing a probability transformer to translate the quantile view to a uniform view, we ensure transformed probabilities are uniformly spread and maintain monotonicity.

3.3.2 Total Calibration Error (TCE) Score Calculation (_calculate_tce_score)
Provides a metric to evaluate and compare the quality of different binning configurations, guiding the iterative adjustment process.

3.3.3 Iterative Binning Adjustment (find_best_binning)
Iteratively finds the best binning configuration by adjusting the bin sizes and boundaries. This process minimizes the TCE score, ensuring the bins are optimally configured for accurate calibration.

3.3.4 Significance Adjustment
Calculates the TCE score by the number of significant deviations between the predicted probabilities and the actual outcomes. For bins with high deviations, significance levels are adjusted dynamically based on the calculated variance and mean, ensuring more precise adjustments.

3.3.5 Normalization
Normalization adjusts predicted probabilities to a common scale, making them comparable across different models or datasets. It preserves the relative differences and ensures monotonicity between probabilities.

3.3.6 Calibration Using Bins
Two options were tested for calibration using bins:

Confidence-based on the variance of the best bins.
Confidence-based on the significance of the bins.
The chosen option achieved a balance between recall and precision, demonstrating the calibrator's ability to improve prediction reliability.

Function Explanations
__init__
Initializes the calibrator with various parameters including the base estimator, cross-validation strategy, number of bins, and significance adjustment factors.

fit
Fits the calibrator using the provided data and labels by training the base estimator on each fold and optimizing the binning configuration.

_check_array
Converts various input data types (e.g., pandas DataFrame, lists) to numpy arrays for consistency.

check_and_print_valid_types
Validates the types of the input data and prints any invalid types for debugging purposes.

_binomial_cdf
Calculates the Cumulative Distribution Function (CDF) for a binomial distribution using log probabilities to improve numerical stability.

_pava_bc
Performs PAV (Pool Adjacent Violators) and ABC (Asymptotic Binomial Calibration) binning with an additional condition on the positive sample ratio.

_calculate_tce_score
Calculates the Total Calibration Error (TCE) score by evaluating the bin deviations.

find_best_binning
Iteratively finds the best binning configuration by adjusting bin sizes and boundaries based on the TCE score.

predict_proba
Uses the calibration parameters obtained during training to adjust the predicted probabilities for better calibration.

plot_pava_bc_statistics
Plots the calibrated probabilities for each bin across all folds using violin plots, along with deviation and positives bar plots, to provide a comprehensive view of the model calibration quality.

_calculate_confidence_intervals
Calculates confidence intervals for the binomial distribution given the number of trials and success probability.
