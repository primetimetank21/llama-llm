# File Dependencies
from pathlib import Path
from datetime import datetime
import tempfile
import json

# Typing Dependencies
from typing import Final, cast

# from io import BufferedRandom
from langchain_core.runnables.base import RunnableSerializable

# LLM Dependencies
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# TTS Dependencies
from gtts import gTTS  # type:ignore
import os

# Input Dependencies
import argparse

# Global Variables
QUERY_SEPARATOR: Final[str] = "\n----------------QUERY SEPARATOR----------------\n"
USER_AI_SEPARATOR: Final[str] = "\n----------------USER AI SEPARATOR----------------\n"


def get_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with the bot!")
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="llama3",
        help="Name of the model to use",
    )
    parser.add_argument(
        "-c", "--context_filename", type=str, default="", help="Context filepath to use"
    )
    return parser.parse_args(args=args)


def speak(filename: str) -> None:
    is_valid_filename: bool = all([isinstance(filename, str), len(filename) > 0])
    if not is_valid_filename:
        raise TypeError("Invalid filename")

    # Play the file
    os.system(f"mpg123 -q {filename}")


def get_context(filename: str, use_context: bool = False) -> str:
    """
    Get the context from a file.

    `filename`: `str` -- The filename to get the context from.\n
    `use_context`: `bool` -- Whether to use the context. Default is `False`.\n
    """
    if not use_context:
        return ""

    # Check if the file exists
    is_valid_filename: bool = all([isinstance(filename, str), len(filename) > 0])
    if not is_valid_filename:
        raise FileExistsError(f"File '{filename}' does not exist")

    with open(filename, "r", encoding="utf-8") as f:
        context_json: dict[str, str | list[dict[str, str]]] = json.load(f)

    context_arr: list[dict[str, str]] = cast(
        list[dict[str, str]], context_json["context"]
    )
    context: str = "".join(
        f"{QUERY_SEPARATOR}User: {question_answer_pair['User']}{USER_AI_SEPARATOR}AI: {question_answer_pair['AI']}{QUERY_SEPARATOR}"
        for question_answer_pair in context_arr
    )

    return context


def save_context(context: str = "") -> None:
    # def save_context(filename:str = "", temp_file: BufferedRandom = None, conversation_over: bool = False) -> None:
    """
    Save the context to a file.

    `context`: `str` -- The context to save.\n
    """
    # Exit if there is no context
    is_valid_context: bool = all([isinstance(context, str), len(context) > 0])
    if not is_valid_context:
        return

    # Create contexts directory
    contexts_dir: Path = Path(Path.cwd(), "contexts")
    contexts_dir.mkdir(parents=True, exist_ok=True)

    # Create file
    timestamp_now: str = datetime.now().strftime("%B %d %Y @ %H:%M:%S")
    timestamp_filename: str = (
        f"context_{timestamp_now.replace(' ', '').replace(':', '_')}.json"
    )
    filename: Path = Path(contexts_dir, timestamp_filename)

    # Format context
    context_lines: list[str] = context.split(QUERY_SEPARATOR)
    question_answer_pairs: list[dict[str, str]] = []

    # Save context
    for line in context_lines:
        if line:
            assert len(line.split(USER_AI_SEPARATOR)) % 2 == 0
            user_line, ai_line = line.split(USER_AI_SEPARATOR)
            user_line, ai_line = (
                user_line.replace("User: ", "", 1),
                ai_line.replace("AI: ", "", 1),
            )

            question_answer_pair: dict[str, str] = {"User": user_line, "AI": ai_line}
            question_answer_pairs.append(question_answer_pair)

    formatted_context: dict[str, str | list[dict[str, str]]] = {
        "started_on": timestamp_now,
        "context": question_answer_pairs,
    }

    # Save context
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(formatted_context, f, indent=4)


def delete_old_context(filename: str) -> None:
    """
    Delete old context files.
    """
    # Check if the file exists
    is_valid_filename: bool = all([isinstance(filename, str), len(filename) > 0])
    if not is_valid_filename:
        raise FileExistsError(f"File '{filename}' does not exist")

    contexts_dir: Path = Path(Path.cwd(), "contexts")
    file_to_delete: Path = Path(contexts_dir, Path(filename).name)
    for file in contexts_dir.glob("*.json"):
        if file == file_to_delete:
            file_to_delete.unlink()
            break


def setup_llm(model_name: str = "llama3") -> RunnableSerializable[dict, str]:
    """
    Setup the LLM.

    `model_name`: `str` -- Name of the model to use. Default is `llama3`.\n
    """

    # TODO: could also provide additional "personality" to the AI
    TEMPLATE: Final[str] = """
    Answer the question below.

    Here is the conversation history: {context}

    Question: {question}

    Answer:
    """
    is_valid_model_name: bool = all([isinstance(model_name, str), len(model_name) > 0])
    if not is_valid_model_name:
        raise Exception(f"Invalid model_name '{model_name}'")

    model: OllamaLLM = OllamaLLM(model=model_name)
    prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(TEMPLATE)
    chain: RunnableSerializable[dict, str] = prompt | model

    return chain


def print_previous_context(context: str = "") -> None:
    """
    Print the previous context.
    """
    # Exit if there is no context
    is_valid_context: bool = all([isinstance(context, str), len(context) > 0])
    if not is_valid_context:
        return

    # Format context
    context_lines: list[str] = context.split(QUERY_SEPARATOR)

    # Print previous context
    for line in context_lines:
        if line:
            assert len(line.split(USER_AI_SEPARATOR)) % 2 == 0
            user_line, ai_line = line.split(USER_AI_SEPARATOR)
            print(f"{user_line}\n{ai_line}\n")


def chat(args: argparse.Namespace) -> None:
    """
    Start chatting with the bot!
    """
    # Setup the LLM
    chain = setup_llm(model_name=args.model_name)

    # Handle the conversation
    CONTINUE_THE_CONVERSATION: bool = True
    use_context: bool = all(
        [isinstance(args.context_filename, str), len(args.context_filename) > 0]
    )
    context: str = get_context(filename=args.context_filename, use_context=use_context)
    # context_filename: str = ""
    # temp_file: BufferedRandom = tempfile.NamedTemporaryFile(encoding="utf-8", delete=False)

    if context:
        print_previous_context(context=context)

    while CONTINUE_THE_CONVERSATION:
        user_input: str = input("User: ")

        if user_input.lower().strip() in ("quit", "exit", "q", "bye"):
            CONTINUE_THE_CONVERSATION = False
            print("AI: Goodbye!")
            break

        # LLM Chatbot's response to the question
        response: str = chain.invoke(input={"context": context, "question": user_input})
        print(f"AI: {response}\n")
        context += f"{QUERY_SEPARATOR}User: {user_input}{USER_AI_SEPARATOR}AI: {response}{QUERY_SEPARATOR}"

        # TTS
        with tempfile.NamedTemporaryFile(
            prefix="tmp_", suffix=".mp3", delete=True
        ) as f:
            tts_response: gTTS = gTTS(text=response, lang="en", slow=False)
            tts_response.save(savefile=f.name)
            speak(f.name)

    save_context(context=context)
    if use_context:
        delete_old_context(filename=args.context_filename)


# TODO: add test cases to check functions
def main(argv: list[str] | None = None) -> None:
    if __name__ == "__main__":
        args: argparse.Namespace = get_args(args=argv)
        chat(args=args)
