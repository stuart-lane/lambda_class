"""
PACKAGE SETUP
"""

import numpy as np
from scipy import stats

"""
LAMBDA CLASS UTIL FUNCTIONS
"""

def kernel_weighting(
    X: np.ndarray, 
    x0: float, 
    bandwidth: float, 
    kernel: str
) -> np.ndarray:

    """Compute kernel weighting matrix"""
    if (kernel.lower() not in ("uniform", "triangular", "epanechnikov")):
        raise ValueError("Kernel argument must be 'uniform', 'triangular', or 'epanechnikov'")

    if (kernel.lower() == "uniform"):
        weights = 0.5 * np.diag((np.abs(X - x0) < bandwidth).astype(float))
    elif (kernel.lower() == "triangular"):
        weights = np.diag(np.sqrt(np.maximum(1 - np.abs((X - x0)/bandwidth), 0)))
    else:
        weights = np.diag(np.sqrt(np.maximum(0.75 * (1 - (np.abs(X - x0)/bandwidth)**2), 0)))
        
    return weights


def _validate_inputs(
    Y: np.ndarray,
    D: np.ndarray, 
    X: np.ndarray, 
    x0: float, 
    bandwidth: float,
    p: int = 1,
    exog: np.ndarray = None, 
    inference: bool = True, 
    vcov_type: str = "nonrobust", 
    tau_0: float = 0,
    alpha: float = 0.05
) -> None:

    """Check for wide range of potential data/parameter errors"""

    # Required parameters that cannot be empty
    required_params = {'Y': Y, 'D': D, 'X': X, 'x0': x0, 'bandwidth': bandwidth, 'p': p}
    none_params = [name for name, val in required_params.items() if val is None]
    if none_params:
        raise ValueError(f"The following parameters cannot be None: {', '.join(none_params)}")
    if any(arr.size == 0 for arr in [Y, D, X]):
        raise ValueError("Y, D and X cannot be empty")
    if exog is not None:
        if exog.size == 0:
           raise ValueError("Exog cannot be empty")

    # Check vector lengths and dimensions    
    if len(Y) != len(D) or len(Y) != len(X):
        raise ValueError("Y, D and X must be the same length")
    if Y.reshape(-1,1).shape[1] != 1 or X.reshape(-1,1).shape[1] != 1 or D.reshape(-1,1).shape[1] != 1:
        raise ValueError("Y, D and X must be n x 1 vectors")
    
    # Check scalars, non-negativity constraints etc.
    if not np.isscalar(x0):
        raise ValueError("x0 must be a scalar")
    if not np.isscalar(bandwidth) or bandwidth < 0:
        raise ValueError("Bandwidth must be a positive scalar")
    if not isinstance(p, (int, np.integer)) or p < 0:
        raise ValueError("p must be a nonnegative integer")
    if not np.isscalar(alpha) and alpha > 0 and alpha < 1:
        raise ValueError("Alpha must be strictly between 0 and 1")
    
    # Ensure sufficient observations
    effective_sample_size = np.sum((np.abs(X - x0) < bandwidth).astype(int))
    minimum_effective_sample_size = 2 * (p + 1) 
    if (effective_sample_size < minimum_effective_sample_size):
        raise ValueError("Insufficient observations to compute estimator")
    
    # Ensure valid covariance-variance matrix type
    if not inference:
        if vcov_type.lower() not in ['nonrobust', 'robust', 'cluster']:
            raise ValueError("vcov_type must be 'nonrobust', 'robust', or 'cluster'")
        
    # Handle tau_0 input
    if tau_0 is None:
        tau_0 = 0.0  # Default to testing against zero

def matrix_lambda_term(
    M: np.ndarray, 
    projection_matrix: np.ndarray, 
    _lambda: float
) -> np.ndarray:

    """Compute inner part of matrix terms"""
    return np.identity(M) - (_lambda * projection_matrix)


"""
LAMBDAFRD MAIN FUNCTION
"""

def lambdaFRD(
    Y: np.ndarray,
    D: np.ndarray, 
    X: np.ndarray, 
    x0: float, 
    bandwidth: float,
    kernel: str = "uniform", 
    Lambda: bool = True, 
    psi: float = 1, 
    p: int = 1,
    inference: bool = True, 
    vcov_type: str = "nonrobust", 
    tau_0: float = None,
    exog: np.ndarray = None, 
    _lambda: float = None, 
    alpha: float = 0.05
) -> tuple:

    """Compute the lambda class estimator with optional inference
    
    ---- Parameters ----

    // Compulsory parameters

    Y : array_like
        Dependent variable
    D : array_like
        Treatment
    X : array_like
        Running variable
    x0 : float
        Cutoff
    bandwidth : float
        Bandwidth used for estimator

    // Optional parameters

    kernel : str, optional
        Kernel weighting function ('uniform', 'triangular' or 'epanechnikov')
        Default is 'uniform'
    Lambda : bool, optional
        Use the Lambda(psi) function for the lambda value
        Default is True
    psi : float, optional
        Value of psi in Lambda(psi) function
        Default is 1 (0 gives standard FRD estimator)
    p : int, optional
        Order of local polynomial
        Default is 1
    inference : bool, optional
        Compute confidence intervals and test statistics
        Default is True
    vcov_type : str, optional
        Type of covariance-variance ('nonrobust', 'robust', 'cluster')
        Default is 'nonrobust
    tau_0 : float, array-like, or None
        Null hypothesis value(s) for testing. 
        - If None: defaults to 0 for all coefficients
        - If scalar: tests all coefficients against this value
        - If array: must match length of coefficients, tests each coefficient 
          against corresponding value
    exog : array_like, optional
        Exogenous variables
    _lambda : float, optional
        Custom value of lambda if not using the Lambda(psi) function
    alpha: float, optional
        1 - alpha level for confidence intervals
        Default is 0.05

    ---- Returns ----

    Tuple
        Container for results

    """

    # Check for errors/warnings
    _validate_inputs(Y, D, X, x0, bandwidth, p, exog, inference, vcov_type, alpha)

    # Housekeeping
    n = len(X)
    Z = (X >= x0).astype(int).reshape(-1, 1)
    X_col = X.reshape(-1, 1)
    D = D.reshape(-1, 1)

    # Construct matrix of polynomials of X
    V = np.ones((n, 1)).astype(int)
    if p >= 1:
        for j in range(1, p+1):
            above_cutoff = Z * ((X_col - x0) ** j)
            below_cutoff = (1 - Z) * ((X_col - x0) ** j)

            V = np.hstack([V, above_cutoff, below_cutoff])

    if exog is not None:
        if exog.ndim == 1:
            exog = exog.reshape(-1, 1)
        V = np.hstack([V, exog])

    # Compute the kernel weights
    kernel_matrix = kernel_weighting(X, x0, bandwidth, kernel)

    weighted_Y = kernel_matrix @ Y
    weighted_V = kernel_matrix @ V
    weighted_D = kernel_matrix @ D
    weighted_Z = kernel_matrix @ Z

    # Drop redundant rows
    indices_to_keep = np.where(weighted_Y != 0)[0]

    Yw = weighted_Y[indices_to_keep]
    Dw = weighted_D[indices_to_keep]
    Vw = weighted_V[indices_to_keep]
    Zw = weighted_Z[indices_to_keep]

    M = len(Yw)

    # Stack instrument and treatments with polynomial matrix
    VDw = np.hstack([Vw, Dw])
    VZw = np.hstack([Vw, Zw])
    izz = np.linalg.pinv(VZw.T @ VZw)
    PZVw = VZw @ izz @ VZw.T

    if (Lambda):
        _lambda = 1 - psi / (M - V.shape[1] - 1)

    projection_matrix = np.identity(M) - PZVw 
    matrix_projection_lambda = matrix_lambda_term(M, projection_matrix, _lambda)

    coeffs_num = VDw.T @ matrix_projection_lambda @ Yw 
    coeffs_den = VDw.T @ matrix_projection_lambda @ VDw

    # Final estimator
    coefficients = (np.linalg.pinv(coeffs_den) @ coeffs_num)
    tau_lambda = coefficients[-1]

    ###################
    #### INFERENCE ####
    ###################

    if not inference:
        return {
            'coefficients': coefficients,
            'tau_lambda': tau_lambda
        }
    
    u_hat = Yw - VDw @ coefficients
    matrix_projection_0 = matrix_lambda_term(M, projection_matrix, 0)

    if vcov_type == "robust":
        # Heteroskedasticity-robust
        u_squared_diag = np.diag(u_hat.flatten() ** 2)
        variance_numerator_term = VDw.T @ PZVw @ u_squared_diag @ PZVw @ VDw
        
    else:
        # Homoskedasticity
        sigma_2_hat = (u_hat.T @ u_hat) / (M - VDw.shape[1])
        variance_numerator_term = VDw.T @ matrix_projection_0 @ VDw
        variance_numerator_term = float(sigma_2_hat) * variance_numerator_term
    
    variance_denominator_term = VDw.T @ matrix_projection_lambda @ VDw
    variance_denominator_term_inverse = np.linalg.pinv(variance_denominator_term)
    
    variance_estimator = (variance_denominator_term_inverse @ 
                         variance_numerator_term @ 
                         variance_denominator_term_inverse)
    
    # T statistic and critical value
    standard_errors = np.sqrt(np.diag(variance_estimator))

    # Convert to numpy array and broadcast appropriately
    tau_0 = np.atleast_1d(tau_0)
    
    # Check dimensions
    if len(tau_0) == 1:
        # Scalar case: broadcast to all coefficients
        tau_0 = np.full(coefficients.shape, tau_0[0])
    elif len(tau_0) != len(coefficients.flatten()):
        raise ValueError(
            f"tau_0 must be scalar or have length {len(coefficients.flatten())}, "
            f"got length {len(tau_0)}"
        )
    else:
        # Array case: reshape to match coefficients
        tau_0 = tau_0.reshape(coefficients.shape)

    t_statistics = (coefficients.flatten() - tau_0) / standard_errors

    df = M - VDw.shape[1]
    
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_statistics), df=df))
    t_critical = stats.t.ppf(1 - alpha/2, df=df)
    
    # Confidence intervals
    margin_of_error = t_critical * standard_errors
    confidence_intervals_lower = coefficients.flatten() - margin_of_error
    confidence_intervals_upper = coefficients.flatten() + margin_of_error
    
    # Treatment effect specific results for easier user access
    tau_se = standard_errors[-1]
    tau_tstat = t_statistics[-1] 
    tau_pvalue = p_values[-1]
    tau_ci = (confidence_intervals_lower[-1], confidence_intervals_upper[-1])
    
    return {
        'coefficients': coefficients,
        'tau_lambda': tau_lambda,
        'standard_errors': standard_errors,
        't_statistics': t_statistics,
        'p_values': p_values,
        'confidence_intervals': list(zip(confidence_intervals_lower, confidence_intervals_upper)),
        'tau_se': tau_se,
        'tau_tstat': tau_tstat,
        'tau_pvalue': tau_pvalue,
        'tau_ci': tau_ci,
        'variance_matrix': variance_estimator,
        'degrees_freedom': df,
        'n_obs': M,
        'bandwidth': bandwidth,
        'alpha': alpha,
    }
