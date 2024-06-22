# pylint: disable=arguments-renamed, line-too-long,pointless-string-statement,unused-argument,unbalanced-tuple-unpacking, unused-variable, unused-import, trailing-whitespace, consider-using-enumerate, undefined-variable, missing-module-docstring, wrong-import-order, missing-function-docstring, invalid-name, superfluous-parens, ungrouped-imports

import sys
import warnings
import numpy as np  # Importing numpy for numerical operations
import scipy
from sklearn import clone  # Importing clone function from sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin  # Importing base classes from sklearn
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, f1_score, precision_recall_curve
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils.validation import check_is_fitted, check_X_y  # Importing validation utilities from sklearn
from sklearn.model_selection import check_cv  # Importing cross-validation and train_test_split utilities
from scipy.special import comb, logsumexp
from scipy.stats import binom, binom_test
import math
import pandas as pd  # Importing pandas for data manipulation
import matplotlib.pyplot as plt  # Importing matplotlib for plotting
from matplotlib.lines import Line2D  # Importing Line2D from matplotlib
from matplotlib.patches import Patch  # Importing Patch from matplotlib
import seaborn as sns
# from scipy.interpolate import interp1d
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

class VATPAVABCCalibrator(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    """
    VATPAVABCCalibrator is a calibration method that uses the PAV algorithm and ABC method
    to adjust predicted probabilities for improved calibration. It combines cross-validation
    with a specified base estimator to achieve this.

    Args:
        base_estimator (BaseEstimator): The base estimator from which the calibrated probabilities are derived.
        cv (int or cross-validation generator): The cross-validation strategy.
        n_bins (int): The number of bins for calibration.
        k (float): The scaling factor for significance adjustments.
        c (float): The constant for significance adjustments.
        lower_bound (float): The lower bound for bin_significances.
        upper_bound (float): The upper bound for bin_significances.
        min_bins (int): The minimum number of bins for the PAV algorithm.
        max_bins (int): The maximum number of bins for the PAV algorithm.
    """
    def __init__(self, base_estimator=None, cv=None, n_bins=10, k=0.5, c=0.1, lower_bound=0.01, upper_bound=0.05, min_bins=10, max_bins=10, min_avg_pos_samples_in_bin=5):
        self.base_estimator = base_estimator  # Initialize the base estimator
        self.cv = cv  # Initialize cross-validation strategy
        self.n_bins = n_bins  # Initialize number of bins
        self.k = k  # Initialize scaling factor for significance adjustments
        self.c = c  # Initialize constant for significance adjustments
        self.lower_bound = lower_bound  # Initialize lower bound for bin_significances
        self.upper_bound = upper_bound  # Initialize upper bound for bin_significances
        self.classes_ = None  # Placeholder for class labels
        self.calibration_params_ = []  # List to store calibration parameters
        self.significance = 0.05  # Initialize significance level
        self.init_bin_size_min = None
        self.min_bins = min_bins  # Initialize minimum number of bins
        self.init_bin_size_max = None
        self.max_bins = max_bins  # Initialize maximum number of bins
        self.max_iterations = 100  # Initialize maximum number of iterations
        self.tolerance = 1e-20  # Initialize tolerance for convergence
        self.debug_ = True
        self.min_avg_pos_samples_in_bin = min_avg_pos_samples_in_bin
        self.size_step=1
        np.set_printoptions(precision=20, threshold=sys.maxsize, linewidth=sys.maxsize)
        self.best_threshold = None


    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the calibrator using the provided data and labels (samples and targets). 

        For each fold:
            The base estimator is trained.
            The calibration parameters are optimized.
        """
        # Do all checks before beginning the training
        X, y = check_X_y(X, y)  # Validate input data and labels
        cv = check_cv(self.cv, y, classifier=True)  # Check cross-validation strategy
        self.classes_ = np.unique(y)  # Get unique class labels
        if len(self.classes_) != 2: 
            raise ValueError("The target variable 'y' must have exactly two unique classes.")

        # Iterate through each fold
        for train_index, test_index in cv.split(X, y):  
            # Train this fold's estimator
            estimator = clone(self.base_estimator) 
            estimator.fit(X[train_index], y[train_index]) 
            probabilities = estimator.predict_proba(X[test_index])[:, 1]  

            # Optimize pava-bc binning
            params = self.find_best_binning(probabilities, y[test_index])

            # Store the fold's final training 
            params['estimator'] = estimator  
            self.calibration_params_.append(params)  

        return self 
    
    def _check_array(self, objects_list):
        rv = []
        for obj in objects_list:
            if isinstance(obj, (pd.Series, pd.DataFrame)):
                rv.append(obj.to_numpy())
            elif isinstance(obj, list):
                rv.append(np.array(obj))
            else:
                rv.append(obj)
        return tuple(rv)
    
    def check_and_print_valid_types(self, data, function_name):
        valid_types = (list, tuple, np.ndarray)  # Define valid types
        for element in data:
            if not isinstance(element, valid_types):
                print(f"Invalid type in {function_name} function: {element} of type {type(element)}")

    def _binomial_cdf(self, k, n, p):
        """
        Calculate the Cumulative Distribution Function (CDF) for a binomial distribution using log probabilities
        to improve numerical stability.
        
        Parameters:
        k (int): The number of successful outcomes.
        n (int): The number of trials.
        p (float): The probability of success on a single trial.

        Returns:
        float: The probability of obtaining k or fewer successes in n trials.

        Examples:
        - `_binomial_cdf(1, 2, 0.5)` calculates the probability of obtaining 1 or fewer successes,
        which would include the outcomes of 0 or 1 successes.
        - `_binomial_cdf(2, 2, 0.5)` considers all possible outcomes (0, 1, or 2 successes),
        essentially equating to a probability of 1 (or 100%).
        """
        if not (0 <= p <= 1):
            raise ValueError(f"Probability p must be between 0 and 1. Received p={p}.")
        if k < 0 or n < 0:
            raise ValueError(f"k and n must be non-negative integers. Received k={k}, n={n}.")
        if n == 0:
            return 1.0 if k >= 0 else 0.0
        if p == 0:
            return 1.0 if k >= n else 0.0
        if p == 1:
            return 1.0 if k >= n else 0.0

        log_cdf = []
        for i in range(min(int(k), n) + 1):
            # Calculate log of binomial coefficient
            log_binom_coeff = math.log(comb(n, i))
            # Calculate log probabilities of i successes and n-i failures
            log_prob_i_successes = log_binom_coeff + i * math.log(p) + (n - i) * math.log(1 - p)
            log_cdf.append(log_prob_i_successes)
        # Use logsumexp to return the exponent of the sum of logs of probabilities to maintain numerical stability
        return math.exp(logsumexp(log_cdf))


    def _pava_bc(self, probabilities: np.ndarray, labels: np.ndarray, min_samples: int, max_samples: int, max_positive_ratio=0.1):
        """
        Performs PAV (Pool Adjacent Violators) and ABC (Asymptotic Binomial Calibration) binning with an additional condition on positive sample ratio.
        """
        order = np.argsort(probabilities)
        sorted_probs = probabilities[order]
        sorted_labels = labels[order]
        total_positives = np.sum(sorted_labels)
        max_positives_per_bin = total_positives * max_positive_ratio

        def _condition(y0, y1, w0, w1):
            condition_positive_cap = (y0 + y1 <= max_positives_per_bin)
            condition_max_samples = (w0 + w1 <= max_samples)
            condition_min_samples = (w0 + w1 <= min_samples)
            condition_monotonicity = (y0 / w0 >= y1 / w1) if w0 != 0 and w1 != 0 else True
            return condition_positive_cap and ( (condition_max_samples and condition_monotonicity) )

        # Bin initialization and merging loop
        count = -1
        iso_y = []
        iso_w = []
        for i in range(len(sorted_probs)):
            iso_y.append(sorted_labels[i])
            iso_w.append(1)
            count += 1
            while count > 0 and _condition(iso_y[count - 1], iso_y[count], iso_w[count - 1], iso_w[count]):
                iso_y[count - 1] += iso_y[count]
                iso_w[count - 1] += iso_w[count]
                iso_y.pop()
                iso_w.pop()
                count -= 1

        # Finalizing bins
        index = np.r_[0, np.cumsum(iso_w)]
        bins = np.r_[0.0, [(sorted_probs[index[j] - 1] + sorted_probs[index[j]]) / 2.0 for j in range(1, len(index) - 1)], 1.0]
        bin_true_positives_count = np.array(iso_y)
        bin_total_count = np.array(iso_w)
        binned_predictions = [sorted_probs[index[j]:index[j + 1]] for j in range(len(index) - 1)]

        return binned_predictions, bin_true_positives_count, bin_total_count, bins


    def _calculate_tce_score(self, binned_predictions, bin_true_positives_count, bin_total_count, bin_significances):
        """
        Calculates the Total Calibration Error (TCE) score.
        """
        number_of_bins = len(bin_true_positives_count)
        bin_deviations = np.zeros(number_of_bins)
        for i in range(number_of_bins):  
            pvals = np.array([self._binomial_cdf(int(bin_true_positives_count[i]), int(bin_total_count[i]), p) for p in binned_predictions[i]]) 
            pvals = np.nan_to_num(pvals)
            bin_deviations[i] = sum(pvals <= bin_significances[i])  # Count significant deviations
        tce_score = 100 * bin_deviations.sum() / bin_total_count.sum()
        return tce_score, bin_deviations


    def find_best_binning(self, probabilities: np.ndarray, labels: np.ndarray):
        """
        Iteratively find the best binning configuration.

        Data Sorting - Sorting the data, this is a common step for algorithms that require ordered data like PAVA.
        Initial Bin Size Estimation - Estimate initial bin sizes based on quantile-like divisions of the data.
        Adaptive Bin Adjustment - Use an adaptive mechanism like PAVA-BC to adjust bin sizes. 
        Looping Mechanism - Implement a looping mechanism that continues to adjust bin sizes until an optimal representation of the data is achieved, according to predefined criteria (e.g., minimum variance within bins, maximum variance between bins).

        """
        best_tce_score = float('inf')
        best_bins_edges = None
        best_bin_mean = None
        best_bins_median = None
        best_bin_variance = None
        best_bin_significances = None
        best_bin_true_positives_count = None
        best_bin_total_count = None
        best_number_of_bins = None
        k_values = np.linspace(0.1, 10.0, 10)
        c_values = np.linspace(0.01, 1.0, 10)

        # Normalize probabilities to the range [0, 1]
        max_prob = np.max(probabilities)
        min_prob = np.min(probabilities)
        probabilities = (probabilities - min_prob) / (max_prob - min_prob)

        # for k in k_values:
        #     for c in c_values:

        for bins in range(self.min_bins, self.max_bins + 1):
            min_samples_per_bin = max(5, labels.shape[0] // (bins * 2)) 
            max_samples_per_bin = min(100000, labels.shape[0] // bins)  
            step_size = (max_samples_per_bin - min_samples_per_bin) // 10 if (max_samples_per_bin - min_samples_per_bin) // 10 > 1 else 1
            # for min_samples in range(min_samples_per_bin, max_samples_per_bin, step_size):
            min_samples = min_samples_per_bin
            for max_samples in range(max_samples_per_bin, min_samples, -step_size):
                # Produce binns
                binned_predictions, bin_true_positives_count, bin_total_count, bins_edges = self._pava_bc(probabilities, labels, min_samples=min_samples, max_samples=max_samples)
                # Calculate variance within each bin
                bin_variance = np.array([np.var(bin_pred) for bin_pred in binned_predictions])
                # Calculate mean within each bin
                bin_mean = np.array([np.mean(bin_pred) for bin_pred in binned_predictions])
                # Calculate median within each bin
                bin_median = np.array([np.median(bin_pred) for bin_pred in binned_predictions])
                # Calculate significance of each bin
                bin_significances = np.clip(self.significance / (1 + self.k * np.sqrt(bin_variance) + self.c * bin_mean), self.lower_bound, self.upper_bound)
                # Loss score
                tce_score, _ = self._calculate_tce_score(binned_predictions, bin_true_positives_count, bin_total_count, bin_significances)
                if tce_score < best_tce_score:
                    best_tce_score = tce_score
                    best_bins_edges = bins_edges
                    best_bin_mean = bin_mean
                    best_bins_median = bin_median
                    best_bin_variance = bin_variance
                    best_min_samples_per_bin = min_samples
                    best_max_samples_per_bin = max_samples
                    best_bin_true_positives_count = bin_true_positives_count
                    best_bin_total_count = bin_total_count
                    best_number_of_bins = bins_edges.shape[0] - 1
                    best_bin_significances = bin_significances
                    
            # Print all variables in a detailed format
            # print("\n\n", "-----"*50, '\n',  "-----"*50)
            # print(f"Checking maximum of {bins} bins")
            # print("-----"*50)
            # print(f"Updated Best TCE Score: {best_tce_score}")
            # print(f"Best number of bins: {best_number_of_bins}")
            # print(f"Best best_bins_edges: {best_bins_edges}")
            # print(f"best best mean: {best_bin_mean}")
            # print(f"best best variance: {best_bin_variance}")
            # print(f"Best best_min_samples_per_bin: {best_min_samples_per_bin}")
            # print(f"Best best_max_samples_per_bin: {best_max_samples_per_bin}")
            # print(f"Best Bin True Positives Count: {best_bin_true_positives_count}")
            # print(f"Best Bin Total Count: {best_bin_total_count}")

        return {'best_number_of_bins': best_number_of_bins,
                'best_bins_edges': best_bins_edges,
                'best_bins_mean': best_bin_mean,
                'best_bins_median': best_bins_median,
                'best_bins_variance': best_bin_variance,
                'best_bins_significance': best_bin_significances,
                'best_bins_empirical_probability': best_bin_true_positives_count / best_bin_total_count,
            }


    def predict_proba(self, X: pd.DataFrame):
        """
        This method uses the calibration parameters obtained during the training process to adjust the predicted probabilities for better calibration. 
        The calibration process involves using bin statistics such as bin means, variances, and sizes to fine-tune the probability estimates.

        Rationalization:
        - Empirical Probabilities:
        Directly using best_bins_empirical_probability from training in the calibration step reflects the true likelihood of outcomes observed during training within each bin.
        This integration allows the calibrated model to approximate the true probabilities observed in the training dataset more closely, leading to improved model accuracy and reliability in prediction.
        
        - Confidence Calculation: 
        Confidence in the calibration is calculated based on bin statistics, specifically the variance and size of the bin.
        Larger bin sizes and lower variances increase confidence because they indicate more robust estimates from the training data.
        Confidence is better aligned as a factor of both variance and bin size, which makes it a dynamic measure adjusting the weight given to empirical versus predicted probabilities.
        Higher confidence gives more weight to the bin mean, while lower confidence gives more weight to the original predictions, providing a balanced approach
        to integrating new predictions with established data trends.
        
        - Normalization:
        The predicted probabilities are normalized to the range [0, 1] based on the minimum and maximum predicted probabilities from the training data. 
        This ensures that the calibration process works uniformly across different ranges of predicted probabilities.

        The method ensures stability and generalizability of the probability estimates by averaging the calibrated probabilities from all cross-validation folds.
        """
        X = self._check_array([X])[0]
        check_is_fitted(self, "calibration_params_")  # Ensure the model has been fit
        proba = np.zeros((X.shape[0], 2))  # Initialize the probability array
        # Aggregate predictions from each fold
        for params in self.calibration_params_:
            estimator = params['estimator']
            predicted_probabilities = np.array(estimator.predict_proba(X)[:, 1])
            best_number_of_bins = params['best_number_of_bins']
            best_bins_edges = params['best_bins_edges']
            best_bins_mean = params['best_bins_mean']
            best_bins_variance = params['best_bins_variance']
            best_bins_median = params['best_bins_median']
            best_bins_significance = params['best_bins_significance']
            best_bins_empirical_probability = params['best_bins_empirical_probability'] 

            # Normalize predicted probabilities to the range [0, 1] based on training data range
            max_pred = np.max(predicted_probabilities)
            min_pred = np.min(predicted_probabilities)
            normalized_predictions = (predicted_probabilities - min_pred) / (max_pred - min_pred)

            # Calibrate the predicted probabilities using the bins
            calibrated_proba = np.zeros(predicted_probabilities.shape[0])
            for j in range(best_number_of_bins):
                bin_mask = (normalized_predictions >= best_bins_edges[j]) & (normalized_predictions < best_bins_edges[j + 1])
                if np.any(bin_mask):
                    confidence =  1 - ( best_bins_variance[j] / ( best_bins_variance[j] + bin_mask.sum() ) ) 
                    calibrated_proba[bin_mask] = (confidence * best_bins_empirical_probability[j] + (1 - confidence) * predicted_probabilities[bin_mask])
            # Handle the last bin edge explicitly
            bin_mask = (normalized_predictions >= best_bins_edges[-1])
            if np.any(bin_mask):
                confidence = best_bins_significance[j] * ( 1 - ( best_bins_variance[-1] / ( best_bins_variance[-1] + bin_mask.sum() ) ) )
                calibrated_proba[bin_mask] = (confidence * best_bins_empirical_probability[-1] + (1 - confidence) * predicted_probabilities[bin_mask])
            # Sum up the probabilities for averaging later
            proba[:, 1] += calibrated_proba / len(self.calibration_params_)
            proba[:, 0] += (1 - calibrated_proba) / len(self.calibration_params_)

        return proba

    def plot_pava_bc_statistics(self, X, y):
        """
        Plots the calibrated probabilities for each bin across all folds using violin plots,
        along with deviation and positives bar plots, to provide a comprehensive view of the model calibration quality.
        """
        # Check and convert data types
        X, y = self._check_array([X, y])

        # Collect predicted probabilities from all calibration models
        all_probs = np.array([params['estimator'].predict_proba(X)[:, 1] for params in self.calibration_params_])
        
        # Average of all model predictions for unadjusted probabilities
        not_adjusted_probabilities = np.mean(all_probs, axis=0)
        
        # Predicted probabilities using the fully adjusted model
        adjusted_probabilities = self.predict_proba(X)[:, 1]

        # Determine the number of bins based on average bins used across models
        number_of_bins = int(np.mean([params['best_number_of_bins'] for params in self.calibration_params_]))
        # Generate intermediate percentiles, excluding the first 0% and last 100% to manually include 0.0 and 1.0 later
        intermediate_percentiles = np.linspace(0, 100, number_of_bins + 1)[1:-1]
        # Calculate bin edges from percentiles of the pooled probabilities
        intermediate_edges = np.percentile(np.concatenate(all_probs), intermediate_percentiles)
        # Combine the edges with 0.0 at the start and 1.0 at the end to ensure full coverage from 0 to 1
        common_bins_edges = np.concatenate(([0.0], intermediate_edges, [1.0]))

        # Initialize lists to collect data for each bin
        adjusted_data = []
        not_adjusted_data = []
        bin_info1 = []  # For first plot (probabilities)
        bin_info2 = []  # For second plot (deviations)
        bin_info3 = []  # For third plot (positives)
        bin_info4 = []  # For fourth plot (calibration)

        # Loop through each pair of edges to define bins
        for j in range(number_of_bins):
            not_adjusted_bin_mask = (not_adjusted_probabilities >= common_bins_edges[j]) & (not_adjusted_probabilities < common_bins_edges[j + 1])
            adjusted_bin_mask = (adjusted_probabilities >= common_bins_edges[j]) & (adjusted_probabilities < common_bins_edges[j + 1])

            not_adjusted_probabilities_in_bin = not_adjusted_probabilities[not_adjusted_bin_mask]
            adjusted_probabilities_in_bin = adjusted_probabilities[adjusted_bin_mask]

            not_adjusted_labels_in_bin = y[not_adjusted_bin_mask]
            adjusted_labels_in_bin = y[adjusted_bin_mask]

            not_adjusted_positive_labels_count = not_adjusted_labels_in_bin.sum()
            adjusted_positive_labels_count = adjusted_labels_in_bin.sum()

            not_adjusted_total_count = not_adjusted_labels_in_bin.shape[0]
            adjusted_total_count = adjusted_labels_in_bin.shape[0]
            
            not_adjusted_empirical_probability = not_adjusted_positive_labels_count / not_adjusted_total_count if not_adjusted_total_count > 0 else 0
            adjusted_empirical_probability = adjusted_positive_labels_count / adjusted_total_count if adjusted_total_count > 0 else 0

            not_adjusted_low, not_adjusted_high = self._calculate_confidence_intervals(not_adjusted_total_count, not_adjusted_empirical_probability)
            adjusted_low, adjusted_high = self._calculate_confidence_intervals(adjusted_total_count, adjusted_empirical_probability)

            # Calculation of before and after calibration deviations
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                not_adjusted_pvals = np.array([binom_test(not_adjusted_positive_labels_count, not_adjusted_total_count, p=p) for p in not_adjusted_probabilities_in_bin]) 
                not_adjusted_deviations = sum((not_adjusted_pvals <= 0.05))
                adjusted_pvals = np.array([binom_test(adjusted_positive_labels_count, adjusted_total_count, p=p) for p in adjusted_probabilities_in_bin]) 
                adjusted_deviations = sum((adjusted_pvals <= 0.05))

            # Save bin information for each plot
            bin_info1.append((j, not_adjusted_low, not_adjusted_high, adjusted_low, adjusted_high))
            bin_info2.append((j, not_adjusted_total_count, not_adjusted_positive_labels_count, not_adjusted_deviations, not_adjusted_probabilities_in_bin))
            bin_info3.append((j, adjusted_total_count, adjusted_positive_labels_count, adjusted_deviations, adjusted_probabilities_in_bin))
            bin_info4.append((j, common_bins_edges[j], common_bins_edges[j + 1]))

            # Collect data for plotting
            if len(adjusted_probabilities_in_bin) > 0:
                adjusted_data.append(pd.DataFrame({
                    'Bin': [j] * len(adjusted_probabilities_in_bin),
                    'Probability': adjusted_probabilities_in_bin,
                    'Empirical Probability': [adjusted_empirical_probability] * len(adjusted_probabilities_in_bin),
                    'Positive Count': [adjusted_positive_labels_count] * len(adjusted_probabilities_in_bin),
                    'Total Count': [adjusted_total_count] * len(adjusted_probabilities_in_bin),
                    'Confidence Low': [adjusted_low] * len(adjusted_probabilities_in_bin),
                    'Confidence High': [adjusted_high] * len(adjusted_probabilities_in_bin),
                    'Deviations': [adjusted_deviations] * len(adjusted_probabilities_in_bin)
                }))
            
            if len(not_adjusted_probabilities_in_bin) > 0:
                not_adjusted_data.append(pd.DataFrame({
                    'Bin': [j] * len(not_adjusted_probabilities_in_bin),
                    'Probability': not_adjusted_probabilities_in_bin,
                    'Empirical Probability': [not_adjusted_empirical_probability] * len(not_adjusted_probabilities_in_bin),
                    'Positive Count': [not_adjusted_positive_labels_count] * len(not_adjusted_probabilities_in_bin),
                    'Total Count': [not_adjusted_total_count] * len(not_adjusted_probabilities_in_bin),
                    'Confidence Low': [not_adjusted_low] * len(not_adjusted_probabilities_in_bin),
                    'Confidence High': [not_adjusted_high] * len(not_adjusted_probabilities_in_bin),
                    'Deviations': [not_adjusted_deviations] * len(not_adjusted_probabilities_in_bin)
                }))

        adjusted_probabilities_data = pd.concat(adjusted_data, ignore_index=True, axis=0)
        not_adjusted_probabilities_data = pd.concat(not_adjusted_data, ignore_index=True, axis=0)

        # Initialize the plot layout
        sns.set_theme(style="whitegrid")
        np.set_printoptions(precision=10, threshold=sys.maxsize, linewidth=sys.maxsize)
        fig, axes = plt.subplots(4, 1, figsize=(40, 80), sharex=False, sharey=False)  # (WIDTH_SIZE, HEIGHT_SIZE)

        # Violin plot for probabilities
        husl_palette = sns.color_palette("hls")
        adjusted_violin_color = husl_palette[5]
        adjusted_probabilities_color = husl_palette[4]
        sns.violinplot(data=adjusted_probabilities_data, x='Bin', y='Probability', ax=axes[0], color=adjusted_violin_color, inner="box", scale='area', join=False)
        sns.stripplot(data=adjusted_probabilities_data, x='Bin', y='Probability', ax=axes[0], color=adjusted_probabilities_color, alpha=0.5)
        max_prob = adjusted_probabilities_data.max()

        not_adjusted_violin_color = husl_palette[3]
        not_adjusted_probabilities_color = husl_palette[2]
        sns.violinplot(data=not_adjusted_probabilities_data, x='Bin', y='Probability', ax=axes[0], color=not_adjusted_violin_color, inner="box", scale='area', join=False, alpha=0.5)
        sns.stripplot(data=not_adjusted_probabilities_data, x='Bin', y='Probability', ax=axes[0], color=not_adjusted_probabilities_color, alpha=0.5)
        max_prob = max(adjusted_probabilities_data['Probability'].max(), not_adjusted_probabilities_data['Probability'].max())

        axes[0].set_ylim(0, max_prob * 1.1)
        axes[0].set_ylabel('Probability')

        # Add horizontal lines for not adjusted and adjusted low and high confidence intervals
        for i, (j, not_adjusted_low, not_adjusted_high, adjusted_low, adjusted_high) in enumerate(bin_info1):
            axes[0].hlines(y=not_adjusted_low, xmin=j-0.4, xmax=j+0.4, color='blue', linestyle='--', linewidth=1)
            axes[0].hlines(y=not_adjusted_high, xmin=j-0.4, xmax=j+0.4, color='blue', linestyle='--', linewidth=1)
            axes[0].hlines(y=adjusted_low, xmin=j-0.4, xmax=j+0.4, color='red', linestyle='--', linewidth=1)
            axes[0].hlines(y=adjusted_high, xmin=j-0.4, xmax=j+0.4, color='red', linestyle='--', linewidth=1)

        # Legend for the violin plot
        legend_elements = [
            Line2D([0], [0], color='blue', linestyle='--', lw=2, label='Not Adjusted High and Low'),
            Line2D([0], [0], color='red', linestyle='--', lw=2, label='Adjusted High and Low'),
            Line2D([0], [0], color=adjusted_probabilities_color, marker='o', linestyle='', markersize=5, label='Adjusted Probabilities'),
            Line2D([0], [0], color=not_adjusted_probabilities_color, marker='o', linestyle='', markersize=5, label='Not Adjusted Probabilities'),
            Line2D([0], [0], color=adjusted_violin_color, marker='s', linestyle='', markersize=5, label='Adjusted Probabilities Violin Boxes'),
            Line2D([0], [0], color=not_adjusted_violin_color, marker='s', linestyle='', markersize=5, label='Not Adjusted Probabilities Violin Boxes')
        ]
        axes[0].legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=8, framealpha=1, borderaxespad=0.5)

        # Deviations bar plot
        flare_palette = sns.color_palette("flare", 2)     
        unique_old = not_adjusted_probabilities_data[['Bin', 'Deviations']].drop_duplicates()
        unique_new = adjusted_probabilities_data[['Bin', 'Deviations']].drop_duplicates()
        unique_all = unique_old.merge(unique_new, on='Bin', how='outer', suffixes=('_Not_Adjusted', '_Adjusted')).fillna(0)
        melted_data = pd.melt(unique_all, id_vars=['Bin'], 
                            value_vars=['Deviations_Not_Adjusted', 'Deviations_Adjusted'],  
                            var_name='Deviation Type', value_name='Deviation')
        sns.barplot(data=melted_data, x='Bin', y='Deviation', hue='Deviation Type', palette=flare_palette,  ax=axes[1])
        axes[1].set_ylabel('Deviation')

        # Legend for the deviations bar plot
        axes[1].legend(
            handles=[Line2D([0], [0], color=flare_palette[0], lw=4), 
                    Line2D([0], [0], color=flare_palette[1], lw=4)],
            labels=["Red: Not Adjusted Deviations", "Yellow: Adjusted Deviations"],
            loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=8, framealpha=1, borderaxespad=0.5
        )

        # Positives bar plot (side-by-side without overlap)
        positives_data = pd.concat([adjusted_probabilities_data.assign(Adjustment='Adjusted'),
                                    not_adjusted_probabilities_data.assign(Adjustment='Not Adjusted')])
        viridis_palette = sns.color_palette("viridis", 2)     
        sns.barplot(data=positives_data, x='Bin', y='Positive Count', hue='Adjustment', palette=viridis_palette, ax=axes[2])
        axes[2].set_ylabel('Positive Count')

        # Annotate the Positive Count bars
        for p in axes[2].patches:
            height = p.get_height()
            if np.isnan(height):
                height = 0
            axes[2].annotate(str(int(height)), 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha = 'center', va = 'center', 
                            xytext = (0, 10), 
                            textcoords = 'offset points')

        # Legend for the positives bar plot
        axes[2].legend(
            handles=[Line2D([0], [0], color=viridis_palette[0], lw=4), 
                    Line2D([0], [0], color=viridis_palette[1], lw=4)],
            labels=["Green: Not Adjusted Positives", "Purple: Adjusted Positives"],
            loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=8, framealpha=1, borderaxespad=0.5
        )

        # Calibration Plot (both adjusted and not adjusted on the same subplot)
        prob_true_adj, prob_pred_adj = calibration_curve(y, adjusted_probabilities, n_bins=number_of_bins, strategy='quantile')
        prob_true_not_adj, prob_pred_not_adj = calibration_curve(y, not_adjusted_probabilities, n_bins=number_of_bins, strategy='quantile')

        sns.lineplot(x=prob_pred_adj, y=prob_true_adj, ax=axes[3], marker='o', linestyle='-', color=sns.hls_palette(h=.5)[0], label='Adjusted Calibration Curve')
        sns.lineplot(x=prob_pred_not_adj, y=prob_true_not_adj, ax=axes[3], marker='o', linestyle='-', color=sns.hls_palette(h=.5)[1], label='Not Adjusted Calibration Curve')
        sns.lineplot(x=[0, 1], y=[0, 1], ax=axes[3], linestyle='--', color='grey', label='Perfect calibration')
        axes[3].set_ylabel('Fraction of positives')
        axes[3].set_xlabel('Mean predicted probability')
        axes[3].grid(True)


        for edge in common_bins_edges:
            axes[3].axvline(x=edge, color='red', linestyle='--', linewidth=1)

        # Set x-axis labels to integers
        axes[3].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Focus on the relevant range for the calibration plot
        axes[3].set_xlim(0, max(prob_pred_adj.max(), prob_pred_not_adj.max())*1.1)
        axes[3].set_ylim(0, max(prob_true_adj.max(), prob_true_not_adj.max())*1.1)

        # Calculate Brier scores for the adjusted and not adjusted probabilities
        brier_score_adj = brier_score_loss(y, adjusted_probabilities)
        brier_score_not_adj = brier_score_loss(y, not_adjusted_probabilities)

        # Legend for the calibration plot
        calibration_handles = [Line2D([0], [0], color=sns.hls_palette(h=.5)[0], lw=4), 
                            Line2D([0], [0], color=sns.hls_palette(h=.5)[1], lw=4), 
                            Line2D([0], [0], color='grey', lw=4, linestyle='--'),
                            Line2D([0], [0], color='red', lw=4, linestyle='--'), 
                            Line2D([0], [0], color='blue', lw=4, linestyle='--')]
        calibration_labels = [
            f"Adjusted Calibration Curve (Brier Score: {brier_score_adj:.6f})",
            f"Not Adjusted Calibration Curve (Brier Score: {brier_score_not_adj:.6f})",
            "Grey: Perfect Calibration",
            "Red: Edges of Bins",
        ]

        axes[3].legend(handles=calibration_handles, labels=calibration_labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=8, framealpha=1, borderaxespad=0.5)
        

          # Main legend for test stats per bin data
        main_legend_labels = [
            f"Bin {j}: Total = {total}, Deviations = {deviation}, Positive Outcomes = {posnum}, Mean = {np.mean(pred):.6f}, Min = {np.min(pred):.6f}, Max = {np.max(pred):.6f}, Std = {np.std(pred):.6f}"
            for j, total, posnum, deviation, pred in (bin_info2 if len(bin_info2) <= 20 else bin_info2[:20])
        ]
        fig.legend(
            handles=[Line2D([0], [0], color='black', marker='*', markersize=10) for _, _, _, _, _ in bin_info2],
            labels=main_legend_labels,
            loc='lower center', ncol=2, fontsize=8, framealpha=1, 
        )
        # Final adjustments
        plt.xticks(rotation=45) 
        plt.tight_layout(pad=5. ,h_pad=15., rect=(0, 0.3, 1, 1))  # (left, bottom, right, top)
        plt.show()


    def _calculate_confidence_intervals(self, n, p):
        """ Calculate confidence intervals for the binomial distribution given number of trials and success probability."""
        if n == 0:
            return 0, 0
        ci_low = binom.ppf(0.05, n, p) / n
        ci_high = binom.ppf(0.95, n, p) / n
        return ci_low, ci_high