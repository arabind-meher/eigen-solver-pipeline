import json


def prompt(matrix: list[list[float]], dimension: int) -> str:
    matrix_str = json.dumps(matrix)

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
                You are a linear algebra expert. Compute the eigenvalues and eigenvectors of the 
                following {dimension}x{dimension} matrix using NumPy and SymPy.

                Matrix: {matrix_str}

                Follow these rules:
                1. Characteristic polynomial coefficients of det(A - λI) = 0, highest degree first.
                   The leading coefficient must be {'-1.0' if dimension % 2 == 1 else '1.0'} for a {dimension}x{dimension} matrix.

                2. Sort eigenvalues in descending order, rounded to 6 decimal places.

                3. For each eigenvalue λ_i, compute the shifted matrix (A - λ_i * I), rounded to 6 decimal places.

                4. Compute the RREF of each shifted matrix.

                5. Normalize each eigenvector so the component with the largest absolute value equals exactly 1.0.
                   Example: if raw vector is [2.0, -4.0, 1.5], largest abs value is 4.0, divide all by 4.0 → [0.5, -1.0, 0.375]
                   Order eigenvectors to match eigenvalue order.

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
