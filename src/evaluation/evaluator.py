import json
import os

import numpy as np

from .metrics import Metrics

METRIC_KEYS = [
    "eigenvalue_mae",
    "eigenvalue_accuracy",
    "eigenvalue_mape",
    "eigenvector_cosine_similarity",
    "characteristic_polynomial_mae",
    "shifted_matrix_mae",
    "rref_frobenius_error",
]


class Evaluator:
    """Evaluates eigen-solver predictions against ground-truth data."""

    def __init__(self):
        self.metrics = Metrics()

    def _get_intermediate_step(self, steps: list[dict], step_name: str):
        for step in steps:
            if step["step"] == step_name:
                return step["value"]
        return None

    def _get_lambda_matrices(self, steps: list[dict], prefix: str) -> list[list[list[float]]]:
        """Collect all matrices whose step name starts with *prefix*, in order."""
        return [step["value"] for step in steps if step["step"].startswith(prefix)]

    def evaluate_sample(self, result: dict, ground_truth: dict) -> dict:
        pred_eigenvalues = result["result"].get("eigenvalues", [])
        true_eigenvalues = ground_truth["result"].get("eigenvalues", [])
        pred_eigenvectors = result["result"].get("eigenvectors", [])
        true_eigenvectors = ground_truth["result"].get("eigenvectors", [])

        pred_poly = self._get_intermediate_step(result["intermediate_steps"], "characteristic_polynomial_coefficients")
        true_poly = self._get_intermediate_step(
            ground_truth["intermediate_steps"], "characteristic_polynomial_coefficients"
        )

        pred_shifted = self._get_lambda_matrices(result["intermediate_steps"], "shifted_matrix_lambda")
        true_shifted = self._get_lambda_matrices(ground_truth["intermediate_steps"], "shifted_matrix_lambda")

        pred_rref = self._get_lambda_matrices(result["intermediate_steps"], "rref_lambda")
        true_rref = self._get_lambda_matrices(ground_truth["intermediate_steps"], "rref_lambda")

        return {
            "id": result["id"],
            "eigenvalue_mae": self.metrics.eigenvalue_mae(pred_eigenvalues, true_eigenvalues),
            "eigenvalue_accuracy": self.metrics.eigenvalue_accuracy(pred_eigenvalues, true_eigenvalues),
            "eigenvalue_mape": self.metrics.eigenvalue_mape(pred_eigenvalues, true_eigenvalues),
            "eigenvector_cosine_similarity": self.metrics.eigenvector_cosine_similarity(
                pred_eigenvectors, true_eigenvectors
            ),
            "characteristic_polynomial_mae": self.metrics.characteristic_polynomial_mae(pred_poly, true_poly),
            "shifted_matrix_mae": self.metrics.shifted_matrix_mae(pred_shifted, true_shifted),
            "rref_frobenius_error": self.metrics.rref_frobenius_error(pred_rref, true_rref),
        }

    def evaluate(self, results_path: str, dataset_path: str, output_path: str = None):
        with open(results_path) as f:
            results = json.load(f)
        with open(dataset_path) as f:
            dataset = json.load(f)

        gt_index = {sample["id"]: sample for sample in dataset}

        all_metrics = []
        for result in results:
            sample_id = result["id"]
            if sample_id not in gt_index:
                print(f"WARNING: {sample_id} not found in dataset, skipping")
                continue
            metrics = self.evaluate_sample(result, gt_index[sample_id])
            all_metrics.append(metrics)
            self._print_sample(metrics)

        self._print_summary(all_metrics)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(all_metrics, f, indent=2)
            print(f"\nMetrics saved to {output_path}")

        return all_metrics

    def _print_sample(self, metrics: dict):
        print(f"{metrics['id']}:")
        print(f"  Eigenvalue MAE:         {metrics['eigenvalue_mae']}")
        print(f"  Eigenvalue Accuracy:    {metrics['eigenvalue_accuracy']}%")
        print(f"  Eigenvalue MAPE:        {metrics['eigenvalue_mape']}%")
        print(f"  Cosine Similarity:      {metrics['eigenvector_cosine_similarity']}")
        print(f"  Char Poly MAE:          {metrics['characteristic_polynomial_mae']}")
        print(f"  Shifted Matrix MAE:     {metrics['shifted_matrix_mae']}")
        print(f"  RREF Frobenius Error:   {metrics['rref_frobenius_error']}")

    def _print_summary(self, all_metrics: list[dict]):
        print("\n--- OVERALL SUMMARY ---")
        for key in METRIC_KEYS:
            values = [m[key] for m in all_metrics if m[key] is not None]
            avg = round(float(np.mean(values)), 6) if values else None
            print(f"{key}: {avg}")
