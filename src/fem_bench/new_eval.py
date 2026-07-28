from typing import Optional

from fem_bench.evaluate_output import *  # noqa: F403
from fem_bench.evaluate_output import (
    _outputs_match,
    _serialize_inputs,
    _serialize_output,
)


def evaluate_function_output_match(
    reference_fcn,
    generated_fcn,
    inputs: list[list],
    atol: float = 1e-8,
    allow_sign_flip_for_output_indices: Optional[list[int]] = None,
) -> tuple[bool, list]:
    """Evaluate outputs using the original shared-exception behavior."""
    detailed_results = []
    all_match = True

    for i, input_args in enumerate(inputs):
        test_result = {
            "test_case": i + 1,
            "inputs": _serialize_inputs(input_args),
            "reference_output": None,
            "generated_output": None,
            "match": False,
            "error": None,
        }

        try:
            ref_out = reference_fcn(*input_args)
            gen_out = generated_fcn(*input_args)

            test_result["reference_output"] = _serialize_output(ref_out)
            test_result["generated_output"] = _serialize_output(gen_out)

            match = _outputs_match(
                ref_out,
                gen_out,
                atol=atol,
                allow_sign_flip_for_output_indices=allow_sign_flip_for_output_indices,
            )
            test_result["match"] = match
            if not match:
                all_match = False
        except Exception as e:
            test_result["error"] = str(e)
            all_match = False

        detailed_results.append(test_result)

    return all_match, detailed_results
