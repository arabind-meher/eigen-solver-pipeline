import json


def prompt(
    matrix: list[list[float]],
    dimension: int,
    intermediate_steps: list[dict],
) -> str:
    matrix_str = json.dumps(matrix)
    steps_str = json.dumps(intermediate_steps, indent=4)

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
                You are a linear algebra expert. The intermediate computation steps below have
                been verified by a symbolic math engine (NumPy/SymPy). Do NOT recompute anything.

                Matrix: {matrix_str}

                ── VERIFIED INTERMEDIATE STEPS ─────────────────────────────────────────────
                {steps_str}
                ────────────────────────────────────────────────────────────────────────────

                Your tasks:
                1. Copy ALL verified intermediate_steps into your response exactly as given.
                   Do not alter any value — polynomial, eigenvalues, shifted matrices, or RREFs.

                2. Use these verified eigenvalues exactly (already sorted descending): {json.dumps(eigenvals)}
                   Do NOT recompute or reorder them.

                3. For each eigenvalue λ_i, extract the eigenvector from its verified RREF:
                   - Identify the free variable column(s)
                   - Set the free variable to 1 and express all other components
                   - Normalize: divide every component by the one with the largest absolute value
                     so the largest absolute value component equals exactly 1.0
                     Example: [2.0, -4.0, 1.5] → divide by 4.0 → [0.5, -1.0, 0.375]
                   - Order eigenvectors to match eigenvalue order

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
