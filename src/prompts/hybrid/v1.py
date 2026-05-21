import json


def prompt(matrix: list[list[float]], dimension: int, eigenvalues: list[float], eigenvectors: list[list[float]]) -> str:
    matrix_str = json.dumps(matrix)
    eigenvalues_str = json.dumps(eigenvalues)
    eigenvectors_str = json.dumps(eigenvectors)

    # Build output schema matching v1/v3 format
    steps_example = [
        {"step": "characteristic_polynomial_coefficients", "value": ["float"] * (dimension + 1)},
        {"step": "eigenvalues", "value": ["float"] * dimension},
    ]
    for i in range(1, dimension + 1):
        steps_example.append(
            {
                "step": f"shifted_matrix_lambda_{i}",
                "value": [["float"] * dimension for _ in range(dimension)],
            }
        )
        steps_example.append(
            {
                "step": f"rref_lambda_{i}",
                "value": [["float"] * dimension for _ in range(dimension)],
            }
        )
    steps_schema = json.dumps(steps_example, indent=4)

    return f"""
                You are a linear algebra expert. The eigenvalues and eigenvectors below have been
                computed by NumPy and SymPy and are verified to be correct. Do NOT recompute them.

                Matrix: {matrix_str}

                ── VERIFIED RESULTS (computed by NumPy/SymPy) ──────────────────────────────────
                Eigenvalues (sorted descending): {eigenvalues_str}
                Eigenvectors (order matches eigenvalues): {eigenvectors_str}
                ────────────────────────────────────────────────────────────────────────────────

                Your tasks:
                1. Use the verified eigenvalues to compute the characteristic polynomial coefficients
                   of det(A - λI) = 0, highest degree first.
                   The leading coefficient must be {'-1.0' if dimension % 2 == 1 else '1.0'} for a {dimension}x{dimension} matrix.

                2. For each eigenvalue λ_i, compute the shifted matrix (A - λ_i * I), rounded to 6 decimal places.

                3. Compute the RREF of each shifted matrix.

                4. Copy the verified eigenvalues and eigenvectors exactly into the result section.
                   Do NOT alter any value.

                Respond ONLY in this exact JSON format with real float values, no explanation, no markdown:
                {{
                    "intermediate_steps": {steps_schema},
                    "result": {{
                        "eigenvalues": [list of {dimension} floats sorted descending, rounded to 6 decimal places],
                        "eigenvectors": [
                            list of {dimension} eigenvectors each with {dimension} floats rounded to 6 decimal places,
                            normalized so the largest absolute value component in each eigenvector is exactly 1.0,
                            order matches eigenvalue order
                        ]
                    }}
                }}

                Important: In the result section only, all floats must be rounded to a maximum of 6 decimal places.
            """
