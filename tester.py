"""
Implements all CS 131-related test logic; is entry-point for testing framework.
"""

import asyncio
import importlib
from os import environ
import sys
import traceback
from operator import itemgetter

from harness import (
    AbstractTestScaffold,
    run_all_tests,
    get_score,
    write_gradescope_output,
)


class TestScaffold(AbstractTestScaffold):
    """Implement scaffold for Brewin' interpreter; load file, validate syntax, run testcase."""

    def __init__(self, interpreter_lib):
        self.interpreter_lib = interpreter_lib

    def setup(self, test_case):
        inputfile, expfile, srcfile = itemgetter("inputfile", "expfile", "srcfile")(
            test_case
        )

        with open(expfile, encoding="utf-8") as handle:
            expected = list(map(lambda x: x.rstrip("\n"), handle.readlines()))

        try:
            with open(inputfile, encoding="utf-8") as handle:
                stdin = list(map(lambda x: x.rstrip("\n"), handle.readlines()))
        except FileNotFoundError:
            stdin = None

        with open(srcfile, encoding="utf-8") as handle:
            program = handle.readlines()

        return {
            "expected": expected,
            "stdin": stdin,
            "program": program,
        }

    def run_test_case(self, test_case, environment):
        expect_failure = itemgetter("expect_failure")(test_case)
        stdin, expected, program = itemgetter("stdin", "expected", "program")(
            environment
        )
        interpreter = self.interpreter_lib.Interpreter(False, stdin, False)
        try:
            interpreter.validate_program(program)
            interpreter.run(program)
        except Exception as exception:  # pylint: disable=broad-except
            if expect_failure:
                error_type, _ = interpreter.get_error_type_and_line()
                received = [f"{error_type}"]
                if received == expected:
                    return 1
                print("\nExpected error:")
                print(expected)
                print("\nReceived error:")
                print(received)

            print("\nException: ")
            print(exception)
            traceback.print_exc()
            return 0

        if expect_failure:
            print("\nExpected error:")
            print(expected)
            print("\nActual output:")
            print(interpreter.get_output())
            return 0

        passed = interpreter.get_output() == expected
        if not passed:
            print("\nExpected output:")
            print(expected)
            print("\nActual output:")
            print(interpreter.get_output())

        return int(passed)


def __generate_test_case_structure(
    cases, directory, category="", expect_failure=False, visible=lambda _: True
):
    return [
        {
            "name": f"{category} | {i}",
            "inputfile": f"{directory}{i}.in",
            "srcfile": f"{directory}{i}.brewin",
            "expfile": f"{directory}{i}.exp",
            "expect_failure": expect_failure,
            "visible": visible(f"test{i}"),
        }
        for i in cases
    ]


def __generate_test_suite(version, successes, failures):
    return __generate_test_case_structure(
        successes,
        f"v{version}/tests/",
        "Correctness",
        False,
    ) + __generate_test_case_structure(
        failures,
        f"v{version}/fails/",
        "Incorrectness",
        True,
    )


def generate_test_suite_v1():
    """wrapper for generate_test_suite for v1"""
    tests = [
        "test_begin1",
        "test_begin2",
        "test_bool_expr",
        "test_compare_bool",
        "test_compare_int",
        "test_compare_null",
        "test_compare_string",
        "test_fields",
        "test_function_call_same_class",
        "test_fwd_call",
        "test_if",
        "test_inputi",
        "test_inputs",
        "test_instantiate_and_call1",
        "test_instantiate_and_call2",
        "test_instantiate_and_return1",
        "test_instantiate_and_return2",
        "test_int_ops",
        "test_nested_calls",
        "test_nothing",
        "test_pass_by_value",
        "test_print_bool",
        "test_print_combo",
        "test_print_int",
        "test_print_string",
        "test_recursion1",
        "test_recursion2",
        "test_return",
        "test_return_exit",
        "test_return_type",
        "test_set_field",
        "test_set_param",
        "test_str_ops",
        "test_while",
    ]
    fails = [
        "test_call_badargs",
        "test_call_invalid_func",
        "test_dup_class",
        "test_dup_field",
        "test_dup_method",
        "test_eval_invalid_var",
        "test_if",
        "test_incompat_operands1",
        "test_incompat_operands2",
        "test_incompat_operands3",
        "test_incompat_operands4",
        "test_instantiate_invalid",
        "test_null_objref",
        "test_return_nothing",
        "test_set_invalid_var",
        "test_while",
    ]
    return __generate_test_suite(1, tests, fails)


def generate_test_suite_v2():
    """wrapper for generate_test_suite for v2"""
    return __generate_test_suite(
        2,
        # [
        #     # "test_return_default2",
        # ],
        # [
        #     # "test_call_badargs",
        #     # "test_incompat_compare2",
        #     # "test_invalid_return_type",
        #     # "test_return_assign_type",
        # ],
        [
            "test_assign_null",
            "test_begin",
            "test_bool_expr",
            "test_compare_bool",
            "test_compare_int",
            "test_compare_null",
            "test_compare_objref",
            "test_compare_objref2",
            "test_compare_string",
            "test_fields",
            "test_function_call_same_class",
            "test_fwd_call",
            "test_if",
            "test_inher1",
            "test_inher2",
            "test_inher3_overload",
            "test_inher4",
            "test_inher5",
            "test_inher6",
            "test_inher7",
            "test_inher8",
            "test_inher9",
            "test_inputi",
            "test_inputs",
            "test_instantiate_and_call",
            "test_instantiate_and_return",
            "test_int_ops",
            "test_let",
            "test_let2",
            "test_let3_empty",
            "test_let_copy",
            "test_let_shadow",
            "test_let_shadow2",
            "test_let_shadow3",
            "test_let_shadow4",
            "test_ll",
            "test_overload1",
            "test_pass_by_value",
            "test_pass_by_value2",
            "test_print_bool",
            "test_print_int",
            "test_print_string",
            "test_return",
            "test_return_default1",
            "test_return_default2",
            "test_return_exit",
            "test_return_me",
            "test_return_null",
            "test_return_type",
            "test_return_type2",
            "test_return_type3",
            "test_set_field",
            "test_set_param",
            "test_str_ops",
            "test_while",
        ],
        [
            "test_bad_type_param",
            "test_call_badargs",
            "test_call_badargs2",
            "test_call_invalid_func",
            "test_dup_class",
            "test_dup_field",
            "test_dup_formal_param",
            "test_dup_method",
            "test_eval_invalid_var",
            "test_if",
            "test_incompat_compare2",
            "test_incompat_operands1",
            "test_incompat_operands2",
            "test_incompat_operands3",
            "test_incompat_return1",
            "test_incompat_return2",
            "test_incompat_return3",
            "test_incompat_return4",
            "test_incompat_return5",
            "test_incompat_types1",
            "test_incompat_types2",
            "test_incompat_types3",
            "test_inher1",
            "test_inher2",
            "test_inher3",
            "test_instantiate_invalid",
            "test_invalid_return_type",
            "test_let",
            "test_let2",
            "test_let3",
            "test_missing_main",
            "test_null_objref",
            "test_poly1",
            "test_return_assign_type",
            "test_set_invalid_var",
            "test_while",
        ]
    )


def generate_test_suite_v3():
    """wrapper for generate_test_suite for v3"""
    tests = [
        # "test_default_field_value",
        # "test_except1",
        # "test_except10",
        # "test_except13",
        # "test_str_ops",
        # "test_template1",
        # "test_template10",
        # "test_template14",
        # "test_template17",  # fix
        # "test_template7",  # fix
        # "test_template8",
        # "test_template9",
        # "test_except_spec",
        # "test_except_inner",
        # "test_except_loop",
        # "test_except_nested_method",
        #####
        # "template_test_3",
        # "test_except1",
        # "test_except13",
        # "test_str_ops",
        # "test_template1",
        # "test_template8",
        # "v_except1",
        # "v_except2",
        # "v_except3",
        # "v_except4",
        # "v_except5",
        # "v_except6",
        # "v_except7",
        # "v_except8",
        # "v_except9",
        # "v_except10",
        # "v_except11",
        # "v_except12",
        # "v_except13",
        # "v_except14",
        # "v_except15",
        # "v_except16",
        # "v_except17",
        # "v_except18",
        # "v_except19",
        # "v_except20",
        # "v_except21",
        # "v_except22",
        # "v_except24",
        # "v_ll",
        # "v_template1",
        # "v_template2",
        # "v_template3",
        # "v_template5",
        # "v_template6",
        # "v_template7",
        # "v_template8",
        # "v_template9",
        # "v_template10",
        # "v_template11",
        # "v_template12",
        # "v_template13",
        # "v_template14",
        # "v_template15",
        # "v_template16",
        # "v_template17",
        # "v_template_test4",
        # "v_test_except25",
    ]
    fails = [
        # "test_except4",
        # "test_incompat_template_types",
        # "test_template11",
        # "test_template12",
        # "test_template13",  # fix
        # "test_template15",
        # "test_template16",
        # "test_template5",  # fix
        # "test_template6",
        # "test_template_invalid_types",
        # "test_template_invalid_types2",
        # "test_except_scope",
        #####
        # "test_except4",
        # "test_incompat_template_types",
        # "test_template5",
        "v_template1",
        "v_template2",
        "v_template3",
        "v_template4",
        "v_template5",
        "v_template6",
        "v_template7",
        "v_template8",
        "v_template9"
        "v_template10",
        "v_template11",
        "v_template12",
        "v_template13",
        "v_template14",
    ]
    return __generate_test_suite(3, tests, fails)


async def main():
    """main entrypoint: argparses, delegates to test scaffold, suite generator, gradescope output"""
    if not sys.argv:
        raise ValueError("Error: Missing version number argument")
    version = sys.argv[1]
    module_name = f"interpreterv{version}"
    interpreter = importlib.import_module(module_name)

    scaffold = TestScaffold(interpreter)

    match version:
        case "1":
            tests = generate_test_suite_v1()
        case "2":
            tests = generate_test_suite_v2()
        case "3":
            tests = generate_test_suite_v3()
        case _:
            raise ValueError("Unsupported version; expect one of 1,2,3")

    results = await run_all_tests(scaffold, tests)
    total_score = get_score(results) / len(results) * 100.0
    print(f"Total Score: {total_score:9.2f}%")

    # flag that toggles write path for results.json
    write_gradescope_output(results, environ.get("PROD", False))


if __name__ == "__main__":
    asyncio.run(main())
