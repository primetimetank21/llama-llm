"""Pytests for main()"""

from typing import cast
import argparse

# import mock  # type:ignore
import pytest
from ai_chatbot import (
    main,
    # chat,
    get_args,
    speak,
    get_context,
    # save_context,
    # delete_old_context,
)


@pytest.mark.parametrize(
    "filename",
    [
        "",
        " ",
        "contexts/test_context.json",
        "contexts/file_that_does_not_exist.json",
        5,
        False,
        True,
        5.0,
        None,
    ],
)
def test_get_context(filename: str) -> None:
    try:
        # Check for proper variable type, non-empty string, and if the file exists
        context, use_context = get_context(filename=filename)
        assert all([isinstance(context, str), isinstance(use_context, bool)])
        return
    except Exception:
        # Check for improper variable type, empty string, and/or if the file doesn't exist
        with pytest.raises((TypeError, FileExistsError)) as exception_info:
            get_context(filename=filename)
    assert exception_info.type is FileExistsError or exception_info.type is TypeError


# TODO: test os.system?
@pytest.mark.parametrize(
    "filename",
    ["", " ", "example_filename1.mp3", 5, 5.0, None, True, False],
)
def test_speak(filename: str) -> None:
    try:
        expected: None = cast(None, speak(filename=filename))
        assert all([isinstance(filename, str), expected is None])
        return
    except Exception:
        with pytest.raises((TypeError, FileExistsError)) as exception_info:
            speak(filename=filename)

    assert exception_info.type is FileExistsError or exception_info.type is TypeError


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
        else main(),
    )

    assert return_val is None
