import os
import sys
import textwrap

from openai import OpenAI


def get_llm_response_chat(
    prompt: str,
    model: str = "gpt-4o",
    max_tokens: int = 100,
    temperature: float = 0.7,
):
    """
    Query the OpenAI ChatCompletion API
    and return the response text.
    """
    # Initialize the client
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
    )
    # Return the first choice's message content
    return response.choices[0].message.content.strip()


def cowsay_ascii(text: str, width: int = 40) -> None:
    """
    Print `text` in a cowsay-style ASCII bubble, followed by the classic ASCII cow.
    """
    # 1. Wrap text to the specified width.
    wrapped_lines = []
    for line in text.splitlines():
        segments = textwrap.wrap(line, width=width)
        if not segments:
            wrapped_lines.append("")  # keep blank lines
        else:
            wrapped_lines.extend(segments)

    if not wrapped_lines:
        wrapped_lines = [""]  # if there's no text, at least one line

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
    print(r"        \   ^__^")
    print(r"         \  (oo)\_______")
    print(r"            (__)\       )\/\ ")
    print(r"                ||----w |")
    print(r"                ||     ||")


def main():
    # 1. Grab the user prompt from command line arguments
    if len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:])
    else:
        # If no arguments, read from stdin or just prompt
        user_prompt = input("What would you like to ask the AI? ")

    # 2. Get response from the LLM
    try:
        llm_response = get_llm_response_chat(user_prompt)
    except Exception as e:
        # If something goes wrong (e.g. missing API key), just show error in ASCII
        error_message = f"Error: {e}"
        cowsay_ascii(error_message)
        sys.exit(1)

    # 3. Print response in cowsay style
    cowsay_ascii(llm_response)
