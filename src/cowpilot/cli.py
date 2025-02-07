import argparse
import os
import sys
import textwrap

# The new v1+ OpenAI Python library
from openai import OpenAI

DEFAULT_COW = r"""
        \   ^__^
         \  (oo)\_______
            (__)\       )\/\
                ||----w |
                ||     ||
""".strip("\n")


def load_cow_file(cow_path: str) -> str:
    """
    Load the contents of a .cow file and return it as a string.
    If something goes wrong, raise a ValueError.
    """
    try:
        with open(cow_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"Could not load .cow file {cow_path}: {e}")


def get_llm_response_chat_stream(
    prompt: str,
    model: str = "gpt-4o",
    max_tokens: int = 100,
    temperature: float = 0.7,
):
    """
    Stream the response from the OpenAI ChatCompletion API
    and return the entire response text after streaming.
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key:
        raise ValueError("Missing OPENAI_API_KEY environment variable.")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful, witty assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,  # <-- Enables token-by-token streaming
    )

    collected_chunks = []
    for chunk in response:
        chunk_message = chunk.choices[0].delta.get("content", "")
        sys.stdout.write(chunk_message)
        sys.stdout.flush()
        collected_chunks.append(chunk_message)

    final_answer = "".join(collected_chunks).strip()
    return final_answer


def cowsay_bubble_and_cow(text: str, cow_art: str, width: int = 40) -> None:
    """
    Print text in a cowsay-style ASCII bubble, then print the ASCII cow.
    """

    # 1. Wrap text to the specified width
    wrapped_lines = []
    for line in text.splitlines():
        segments = textwrap.wrap(line, width=width)
        if not segments:
            # Keep blank lines
            wrapped_lines.append("")
        else:
            wrapped_lines.extend(segments)

    if not wrapped_lines:
        # Ensure at least one line
        wrapped_lines = [""]

    max_length = max(len(line) for line in wrapped_lines)

    # 2. Build the bubble borders
    top_border = "  " + "_" * (max_length + 2)
    bottom_border = "  " + "-" * (max_length + 2)

    # 3. Print the bubble
    print(top_border)
    for line in wrapped_lines:
        print(f"< {line.ljust(max_length)} >")
    print(bottom_border)

    # 4. Print the custom ASCII cow
    print(cow_art)


def main():
    parser = argparse.ArgumentParser(
        description="Streaming cowsay with optional .cow file."
    )
    parser.add_argument("--cow", type=str, help="Path to a custom .cow file.")
    parser.add_argument("prompt", nargs="*", help="Prompt to send to the LLM.")
    args = parser.parse_args()

    # 1. Determine user prompt
    if args.prompt:
        user_prompt = " ".join(args.prompt)
    else:
        # If no prompt on command line, ask interactively
        user_prompt = input("What would you like to ask the AI? ")

    # 2. Load custom cow file if provided
    if args.cow:
        try:
            cow_art = load_cow_file(args.cow)
        except ValueError as e:
            # If we can't load the cow file, show an error in cowsay style
            cowsay_bubble_and_cow(str(e), DEFAULT_COW)
            sys.exit(1)
    else:
        # Fallback to the default cow ASCII
        cow_art = DEFAULT_COW

    # 3. Get response from the LLM in a streaming fashion
    try:
        llm_response = get_llm_response_chat_stream(user_prompt)
    except Exception as e:
        # If something goes wrong, show error in cowsay style
        cowsay_bubble_and_cow(f"Error: {e}", cow_art)
        sys.exit(1)

    # 4. Print a blank line (just for spacing) and then do cowsay
    print()
    cowsay_bubble_and_cow(llm_response, cow_art)


if __name__ == "__main__":
    main()
