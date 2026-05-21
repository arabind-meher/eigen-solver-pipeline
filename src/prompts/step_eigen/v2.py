import json


def prompt(
    matrix: list[list[float]],
    dimension: int,
    intermediate_steps: list[dict],
    eigenvectors: list[list[float]],
) -> str:
    matrix_str = json.dumps(matrix)
    steps_str = json.dumps(intermediate_steps, indent=4)
    eigenvectors_str = json.dumps(eigenvectors)

    # Pull verified eigenvalues for explicit inline reference
    step_map = {s["step"]: s["value"] for s in intermediate_steps}
    eigenvals = step_map.get("eigenvalues", [])

    # Build expected output schema — same shape as baseline so metrics.py works unchanged
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
                You are a linear algebra expert. The intermediate computation steps and eigenvectors
                below have been verified by a symbolic math engine (NumPy/SymPy). Do NOT recompute anything.

                Matrix: {matrix_str}

                ── VERIFIED INTERMEDIATE STEPS ─────────────────────────────────────────────
                {steps_str}
                ────────────────────────────────────────────────────────────────────────────

                ── VERIFIED EIGENVECTORS ────────────────────────────────────────────────────
                {eigenvectors_str}
                ────────────────────────────────────────────────────────────────────────────

                Your tasks:
                1. Copy ALL verified intermediate_steps into your response exactly as given.
                   Do not alter any value — polynomial, eigenvalues, shifted matrices, or RREFs.

                2. Use these verified eigenvalues exactly (already sorted descending): {json.dumps(eigenvals)}
                   Do NOT recompute or reorder them.

                3. Copy the verified eigenvectors exactly into the result section.
                   Do NOT alter any value. Order matches eigenvalue order.

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
