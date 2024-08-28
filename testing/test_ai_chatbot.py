"""Pytests for main()"""

from typing import cast, Any
import argparse

# import mock  # type:ignore
import pytest
from ai_chatbot import (
    main,
    # chat,
    get_args,
    speak,
    # get_context,
    # save_context,
    # delete_old_context,
)


@pytest.mark.parametrize(
    "filename, bad_filename",
    [
        ("example_filename1.mp3", 5),
        ("example_filename2.mp3", ""),
        ("example_filename3.mp3", None),
        ("example_filename4.mp3", 5.0),
    ],
)
def test_speak(filename: str, bad_filename: Any) -> None:
    # Valid filename
    valid_case: None = cast(None, speak(filename=filename))
    assert valid_case is None

    # Invalid filename
    try:
        speak(filename=bad_filename)
        assert False
    except TypeError:
        assert True


@pytest.mark.parametrize(
    "model_arg_flag, model_name",
    [("-m", "llama3"), ("--model_name", "llama3"), ("", "")],
)
def test_get_args_with_model_name(model_arg_flag: str, model_name: str) -> None:
    is_valid_model_name: bool = all([isinstance(model_name, str), len(model_name) > 0])
    is_valid_model_arg_flag: bool = all(
        [isinstance(model_arg_flag, str), len(model_arg_flag) > 0]
    )
    args: argparse.Namespace = (
        get_args(args=[model_arg_flag, model_name])
        if all([is_valid_model_arg_flag, is_valid_model_name])
        else get_args(args=[])
    )
    assert args.model_name == "llama3"


@pytest.mark.parametrize(
    "model_arg_flag, model_name, context_arg_flag, context_filename",
    [
        ("-m", "llama3", "-c", "contexts/test_context.json"),
        ("--model_name", "llama3", "--context_filename", "contexts/test_context.json"),
        ("", "", "--context_filename", "contexts/test_context.json"),
        ("--model_name", "llama3", "", ""),
        ("", "", "", ""),
    ],
)
def test_main(
    model_arg_flag: str, model_name: str, context_arg_flag: str, context_filename: str
) -> None:
    is_valid_model_name: bool = all([isinstance(model_name, str), len(model_name) > 0])
    is_valid_model_arg_flag: bool = all(
        [isinstance(model_arg_flag, str), len(model_arg_flag) > 0]
    )
    is_valid_context_arg_flag: bool = all(
        [isinstance(context_arg_flag, str), len(context_arg_flag) > 0]
    )
    is_valid_context_filename: bool = all(
        [isinstance(context_filename, str), len(context_filename) > 0]
    )

    return_val: None = cast(
        None,
        main(argv=[model_arg_flag, model_name, context_arg_flag, context_filename])
        if any(
            [
                is_valid_model_arg_flag,
                is_valid_model_name,
                is_valid_context_arg_flag,
                is_valid_context_filename,
            ]
        )
        else main(argv=[]),
    )

    assert return_val is None
