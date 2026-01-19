import importlib
import subprocess
import sys

def ensure_package(pkg_name):
    """Utility function to ensure package is installed"""
    try:
        return importlib.import_module(pkg_name)
    except ModuleNotFoundError:
        print(f"Package '{pkg_name}' not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
        return importlib.import_module(pkg_name)

# Standard packages
import numpy as np
import pandas as pd
rdrobust_module = ensure_package("rdrobust")
rdbwselect = rdrobust_module.rdbwselect
rdrobust = rdrobust_module.rdrobust

"""
APPLICATION SPECIFIC FUNCTIONS
"""

def print_output(
    estimator: pd.Series, 
    label: str
) -> None:

    """Print formatted output"""
    estimator_result = f"{label}: {estimator['tau_lambda']:.2f}"
    confidence_interval = f"CI: [{estimator['tau_ci'][0]:.2f}, {estimator['tau_ci'][1]:.2f}]"

    print(f"{estimator_result}:  {confidence_interval}")


compute_function = False
if not compute_function:
    def calculate_optimal_bw(df, y_scaled, cutoff, bw_type="mserd"):
        """Compute optimal bandwidth from rdbwselect"""
        bw = rdbwselect(
            y=y_scaled,
            x=df['cohsize'],
            c=cutoff,
            covs=df['tipuach'],
            fuzzy=df['classize'],
            bwselect=bw_type,
            masspoints='off'
        )

        optimal_bw = bw.bws.loc[bw_type, "h (left)"]
        return optimal_bw


def rdrobust_estimation_inference(
    Y: np.ndarray, 
    X: np.ndarray, 
    cutoff: float,  
    D: np.ndarray,
    bandwidth: float,
    W: np.ndarray = None
) -> tuple:

    """Compute standard estimator and bias-corrected confidence interval"""
    bw = rdrobust(
        y=Y,
        x=X,
        c=cutoff,
        covs=W,
        fuzzy=D,
        h=bandwidth,
        kernel='triangular',
        masspoints='off'
    )

    frd_estimator = bw.coef['Coeff'].iloc[0]
    bc_estimator = bw.coef['Coeff'].iloc[2]
    frd_ci_lower = bw.ci['CI Lower'].iloc[0]
    frd_ci_upper = bw.ci['CI Upper'].iloc[0]
    bc_ci_lower = bw.ci['CI Lower'].iloc[2]
    bc_ci_upper = bw.ci['CI Upper'].iloc[2]

    return {
        'frd_estimator': frd_estimator,
        'bc_estimator': bc_estimator,
        'frd_ci_lower': frd_ci_lower,
        'frd_ci_upper': frd_ci_upper,
        'bc_ci_lower': bc_ci_lower,
        'bc_ci_upper': bc_ci_upper
    }


def format_rdrobust_results(
    estimator: float,
    ci_lower: float, 
    ci_upper: float, 
    label: str
) -> tuple:
    
    """Format output from rdrobust package"""
    ci = f"CI: [{ci_lower:.2f}, {ci_upper:.2f}]"
    print(f"{label}: {estimator:.2f}, {ci}")


def standardise(
    data: float
) -> float:

    """Manually standardize data (mean=0, std=1)"""
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # ddof=1 for sample std
    return (data - mean) / std