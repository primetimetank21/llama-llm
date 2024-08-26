# File Dependencies
from pathlib import Path
from datetime import datetime
import tempfile

# Typing Dependencies
from typing import Final

# from io import BufferedRandom
from langchain_core.runnables.base import RunnableSerializable

# LLM Dependencies
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# TTS Dependencies
from gtts import gTTS  # type:ignore
import os


def speak(filename: str) -> None:
    is_valid_filename: bool = all([isinstance(filename, str), len(filename) > 0])
    if not is_valid_filename:
        raise Exception("Invalid filename")

    # Play the file
    os.system(f"mpg123 -q {filename}")





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

    # Check if the conversation is over.
    # If it is, save the content from temp_file to filename AND delete temp_file.
    # If it isn't, save the context to temp_file

    # Create contexts directory
    contexts_dir: Path = Path(Path.cwd(), "contexts")
    contexts_dir.mkdir(parents=True, exist_ok=True)

    # Create filename
    timestamp_now: str = datetime.now().strftime("%Y%m%d-%H%M%S")
    timestamp_filename: str = f"context_{timestamp_now}.txt"
    filename: Path = Path(contexts_dir, timestamp_filename)

    # Save context
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Start of the context from {timestamp_now}:\n\n")
        f.write(context.strip())


def setup_llm(model_name: str = "llama3") -> RunnableSerializable[dict, str]:
    """
    Setup the LLM.

    `model_name`: `str` -- Name of the model to use. Default is `llama3`.\n
    """

    TEMPLATE: Final[str] = """
    Answer the question below.

    Here is the conversation history: {context}

    Question: {question}

    Answer:
    """

    model: OllamaLLM = OllamaLLM(model=model_name)
    prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(TEMPLATE)
    chain: RunnableSerializable[dict, str] = prompt | model

    return chain


def chat() -> None:
    """
    Start chatting with the bot!
    """
    # Setup the LLM
    chain = setup_llm("llama3")

    # Handle the conversation
    CONTINUE_THE_CONVERSATION: bool = True
    context: str = ""
    # context_filename: str = ""
    # temp_file: BufferedRandom = tempfile.NamedTemporaryFile(encoding="utf-8", delete=False)

    while CONTINUE_THE_CONVERSATION:
        user_input: str = input("User: ")

        if user_input.lower().strip() in ("quit", "exit", "q", "bye"):
            CONTINUE_THE_CONVERSATION = False
            print("Bot: Goodbye!")
            break
        # elif user_input.lower() in ("save", "s"):
        #     print("Bot: Saving the context")
        #     context_filename = save_context(temp_file=temp_file, conversation_over=not CONTINUE_THE_CONVERSATION)
        #     continue

        # LLM Chatbot's response to the question
        response: str = chain.invoke(input={"context": context, "question": user_input})
        print(f"Bot: {response}")
        context += f"\nUser: {user_input}\nAI: {response}\n"

        # TTS
        with tempfile.NamedTemporaryFile(
            prefix="tmp_", suffix=".mp3", delete=True
        ) as f:
            tts_response: gTTS = gTTS(text=response, lang="en", slow=False)
            tts_response.save(savefile=f.name)
            speak(f.name)

    save_context(context=context)
    # save_context(filename=context_filename, temp_file=temp_file, conversation_over=not CONTINUE_THE_CONVERSATION)


# TODO: add CLI args in order to fetch a specific context (in order to continue conversation)
# TODO: add test cases to check functions
def main() -> None:
    if __name__ == "__main__":
        chat()


main()
