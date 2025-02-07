import argparse
import os
import sys
import textwrap

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

    # 4. Print the ASCII cow
    print(cow_art)


def main():
    parser = argparse.ArgumentParser(
        description="Streaming cowsay with optional .cow file."
    )
    parser.add_argument("-c", "--cow", type=str, help="Path to a custom .cow file.")
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
        cow_art = DEFAULT_COW

    # 3. Stream the AI response while continuously re-drawing the cow and bubble
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if not client.api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable.")

        system_message = """You are the terminal-based cowsay program upgraded with AI. 
You are witty, rowdy, and a cow. When you aren't entertaining people in the terminal, 
you are protecting the most important cow in the whole world, Gemma Cow.
"""

        # Create the streaming ChatCompletion
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=200,
            temperature=0.7,
            stream=True,  # stream token-by-token
        )

        partial_text = ""

        # Clear screen and show initial (empty) speech bubble and cow
        # so the cow is visible from the start
        print("\x1b[2J\x1b[H", end="")  # Clear the screen, move cursor to top
        cowsay_bubble_and_cow("", cow_art)

        # Iterate over streaming chunks
        for chunk in response:
            chunk_message = chunk.choices[0].delta.content
            if chunk_message:
                partial_text += chunk_message

                # Clear screen and re-draw bubble + cow with updated partial text
                print("\x1b[2J\x1b[H", end="")
                cowsay_bubble_and_cow(partial_text, cow_art)

                # Optionally add a tiny sleep to reduce flicker
                # time.sleep(0.02)

        # Optionally, print a final newline
        print()

    except Exception as e:
        # If something goes wrong, show error in cowsay style
        cowsay_bubble_and_cow(f"Error: {e}", cow_art)
        sys.exit(1)


if __name__ == "__main__":
    main()
