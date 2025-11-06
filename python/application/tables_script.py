print("Packages needed for script are loading, please wait... \n\n")
import importlib
import subprocess
import sys
import os

from application_utils import *
from lambda_class_functions import *

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
from sklearn.preprocessing import StandardScaler
rdrobust_module = ensure_package("rdrobust")
rdbwselect = rdrobust_module.rdbwselect
rdrobust = rdrobust_module.rdrobust

"""
USER CONFIGURATION/CUSTOMISATION
"""
SHOW_OUTPUT = True
SAVE_OUTPUT_TO_CSV = True
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DESTINATION = os.path.join(SCRIPT_DIR, "output/anglavy99_output.csv")
DATA_FILE = os.path.join(SCRIPT_DIR, "final4.csv")

"""
DATA SETUUP
"""
tests = ["verb", "math"]
cutoffs = [40, 80, 120]
bandwidths = list(range(6, 20, 2))

row_width = 30 # Prints "=" * row_width in output

results_dataframe_columns = [
    "test", "cutoff", "bandwidth", "n",
    "iv_coeff", "frd_coeff", "bc_robust_coeff",
    "lambda_1_coeff", "lambda_4_coeff",
    "iv_CI_lower", "iv_CI_upper",
    "frd_CI_lower", "frd_CI_upper",
    "bc_robust_CI_lower", "bc_robust_CI_upper",
    "lambda_1_CI_lower", "lambda_1_CI_upper",
    "lambda_4_CI_lower", "lambda_4_CI_upper"
]
"""
MAIN ANALYSIS LOOP
"""

def main():

    if SAVE_OUTPUT_TO_CSV:
        df_results = pd.DataFrame(columns=results_dataframe_columns)

    df = pd.read_csv(DATA_FILE)

    for test in tests:

        if SHOW_OUTPUT:
            print("=" * row_width)
            print(f"{test.upper()} SCORES")
            print("=" * row_width)

        test_column = "avg" + test
        
        for cutoff in cutoffs:

            if SHOW_OUTPUT:
                print("\n")
                print("=" * row_width)
                print(f"CUTOFF: {cutoff}")
                print("=" * row_width + "\n")

            # Optimal bandwidth calculations - reference only, not used in calculations
            scaler = StandardScaler()
            y_scaled = scaler.fit_transform(df[test_column].to_numpy().reshape(-1, 1)).flatten()
    
            mse_bw = calculate_optimal_bw(df, y_scaled, cutoff, bw_type="mserd")
            cov_bw = calculate_optimal_bw(df, y_scaled, cutoff, bw_type="cerrd")

            for bandwidth in bandwidths:         
                df["effective_sample"] = (np.abs((df['cohsize'] - cutoff) / bandwidth) < 1).astype(int)
                filtered_sample = df[df['effective_sample'] == 1]

                scaler = StandardScaler()
                Y = scaler.fit_transform(filtered_sample[test_column].to_numpy().reshape(-1, 1)).flatten()

                # Convert other variables to numpy arrays
                D = filtered_sample['classize'].to_numpy().astype(float)
                X = filtered_sample['cohsize'].to_numpy().astype(float) 
                W = filtered_sample['tipuach'].to_numpy().astype(float)

                # Compute estimators and confidence intervals
                iv_results = lambda_class(Y, D, X, x0 = cutoff, bandwidth = bandwidth, exog = W, kernel="uniform", p=1,
                                        Lambda=True, psi=0, tau_0=0, inference=True, vcov_type="robust")
                lambda_1_results = lambda_class(Y, D, X, x0 = cutoff, bandwidth = bandwidth, exog = W, kernel="uniform", p=1,
                                                Lambda=True, psi=1, tau_0=0, inference=True, vcov_type="robust")
                lambda_4_results = lambda_class(Y, D, X, x0 = cutoff, bandwidth = bandwidth, exog = W, kernel="uniform", p=1,
                                                Lambda=True, psi=4, tau_0=0, inference=True, vcov_type="robust")
                rdrobust_results = rdrobust_estimation_inference(Y, X, cutoff, D, bandwidth, W)
                
                # Print output
                if SHOW_OUTPUT:
                    print(f"Bandwidth: {bandwidth} N: {lambda_1_results['n_obs']}")
                    print("-" * row_width)
                    print(f"MSE optimal bandwidth: {mse_bw:.2f}")
                    print(f"CER optimal bandwidth: {cov_bw:.2f}")
                    print("-" * row_width)
                    print_output(iv_results, 'IV')
                    print_output(lambda_1_results, 'L1')
                    print_output(lambda_4_results, 'L4')
                    format_rdrobust_results(rdrobust_results['frd_estimator'], 
                                            rdrobust_results['frd_ci_lower'],
                                            rdrobust_results['frd_ci_upper'], 'FRD')
                    format_rdrobust_results(rdrobust_results['bc_estimator'], 
                                            rdrobust_results['bc_ci_lower'],
                                            rdrobust_results['bc_ci_upper'], 'BC')
                    print("=" * row_width)       

                # Save output
                if SAVE_OUTPUT_TO_CSV:
                    row = {
                        "test": test,
                        "cutoff": cutoff,
                        "bandwidth": bandwidth,
                        "n": lambda_1_results['n_obs'],
                        "iv_coeff": iv_results['tau_lambda'],
                        "frd_coeff": rdrobust_results['frd_estimator'],
                        "bc_robust_coeff": rdrobust_results['bc_estimator'],
                        "lambda_1_coeff": lambda_1_results['tau_lambda'],
                        "lambda_4_coeff": lambda_4_results['tau_lambda'],
                        "iv_CI_lower": iv_results['tau_ci'][0],
                        "iv_CI_upper": iv_results['tau_ci'][1],
                        "frd_CI_lower": rdrobust_results['frd_ci_lower'],
                        "frd_CI_upper": rdrobust_results['frd_ci_upper'],
                        "bc_robust_CI_lower": rdrobust_results['bc_ci_lower'],
                        "bc_robust_CI_upper": rdrobust_results['bc_ci_upper'],
                        "lambda_1_CI_lower": lambda_1_results['tau_ci'][0],
                        "lambda_1_CI_upper": lambda_1_results['tau_ci'][1],
                        "lambda_4_CI_lower": lambda_4_results['tau_ci'][0],
                        "lambda_4_CI_upper": lambda_4_results['tau_ci'][1]
                    }

                    df_results = pd.concat([df_results, pd.DataFrame([row])], ignore_index=True)

    # Print output and save (creating dir if required)
    if SAVE_OUTPUT_TO_CSV:
        output_dir = os.path.dirname(OUTPUT_DESTINATION)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df_results.to_csv(OUTPUT_DESTINATION, index=False)
    if SHOW_OUTPUT:
        print(f"Successfully saved output to '{OUTPUT_DESTINATION}'")

if __name__ == "__main__":
    main()