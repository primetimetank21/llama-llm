# Typing Dependencies
from typing import Final
from langchain_core.runnables.base import RunnableSerializable

# LLM Dependencies
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


def chat() -> None:
    TEMPLATE: Final[str] = """
    Answer the question below.

    Here is the conversation history: {context}

    Question: {question}

    Answer:
    """

    # Setup the LLM
    model: OllamaLLM = OllamaLLM(model="llama3")
    prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(TEMPLATE)
    chain: RunnableSerializable[dict, str] = prompt | model

    # Handle the conversation
    CONTINUE_THE_CONVERSATION: bool = True
    context: str = ""

    while CONTINUE_THE_CONVERSATION:
        user_input: str = input("User: ")

        if user_input.lower() in ("quit", "exit", "q"):
            CONTINUE_THE_CONVERSATION = False
            print("Bot: Goodbye!")
            break

        response: str = chain.invoke(input={"context": context, "question": user_input})
        print(f"Bot: {response}")
        context += f"\nUser: {user_input}\nAI: {response}\n"


def main() -> None:
    if __name__ == "__main__":
        chat()


main()
