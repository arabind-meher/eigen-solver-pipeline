import numpy as np
from scipy.optimize import linear_sum_assignment


class Metrics:
    """Computes evaluation metrics for eigen-solver predictions."""

    def _match_eigenvalues(self, pred: list[float], true: list[float]) -> tuple[list[float], list[float]]:
        """Match predicted eigenvalues to true eigenvalues using optimal assignment."""
        cost = np.abs(np.array(pred)[:, None] - np.array(true)[None, :])
        row_ind, col_ind = linear_sum_assignment(cost)
        matched_pred = [pred[i] for i in row_ind]
        matched_true = [true[j] for j in col_ind]
        return matched_pred, matched_true

    def eigenvalue_mae(self, pred: list[float], true: list[float]) -> float:
        matched_pred, matched_true = self._match_eigenvalues(pred, true)
        return float(np.mean(np.abs(np.array(matched_pred) - np.array(matched_true))))

    def eigenvalue_accuracy(self, pred: list[float], true: list[float], tolerance: float = 0.0001) -> float:
        matched_pred, matched_true = self._match_eigenvalues(pred, true)
        correct = sum(1 for p, t in zip(matched_pred, matched_true) if abs(p - t) <= tolerance)
        return round(correct / len(true) * 100, 6)

    def eigenvalue_mape(self, pred: list[float], true: list[float]) -> float:
        matched_pred, matched_true = self._match_eigenvalues(pred, true)
        errors = [abs(p - t) / abs(t) * 100 for p, t in zip(matched_pred, matched_true) if abs(t) > 1e-10]
        return round(float(np.mean(errors)) if errors else 0.0, 6)

    def eigenvector_cosine_similarity(self, pred: list[list[float]], true: list[list[float]]) -> float:
        similarities = []
        for p, t in zip(pred, true):
            p_arr, t_arr = np.array(p, dtype=float), np.array(t, dtype=float)
            norm_p, norm_t = np.linalg.norm(p_arr), np.linalg.norm(t_arr)
            if norm_p < 1e-10 or norm_t < 1e-10:
                continue
            similarities.append(abs(np.dot(p_arr, t_arr) / (norm_p * norm_t)))
        return round(float(np.mean(similarities)) if similarities else 0.0, 6)

    def characteristic_polynomial_mae(self, pred: list[float], true: list[float]) -> float | None:
        if not pred or not true:
            return None
        min_len = min(len(pred), len(true))
        return round(float(np.mean(np.abs(np.array(pred[:min_len]) - np.array(true[:min_len])))), 6)

    def shifted_matrix_mae(
        self, pred_matrices: list[list[list[float]]], true_matrices: list[list[list[float]]]
    ) -> float | None:
        """Average element-wise MAE across all shifted matrices (one per eigenvalue)."""
        if not pred_matrices or not true_matrices:
            return None
        maes = [float(np.mean(np.abs(np.array(p) - np.array(t)))) for p, t in zip(pred_matrices, true_matrices)]
        return round(float(np.mean(maes)), 6)

    def rref_frobenius_error(
        self, pred_matrices: list[list[list[float]]], true_matrices: list[list[list[float]]]
    ) -> float | None:
        """Average Frobenius norm error across all RREF matrices (one per eigenvalue)."""
        if not pred_matrices or not true_matrices:
            return None
        errors = [float(np.linalg.norm(np.array(p) - np.array(t), "fro")) for p, t in zip(pred_matrices, true_matrices)]
        return round(float(np.mean(errors)), 6)
