from datetime import datetime
from pathlib import Path

import numpy as np
from openai_harmony import (
    Conversation,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
    SystemContent,
    load_harmony_encoding,
)

from engine import Engine

enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def find_common_prefix_length(cached_tokens: list[int], new_tokens: list[int]) -> int:
    """Find the length of common prefix between cached and new tokens."""
    min_len = min(len(cached_tokens), len(new_tokens))
    for i in range(min_len):
        if cached_tokens[i] != new_tokens[i]:
            return i
    return min_len


def get_input_tokens(messages: list[Message]) -> list[int]:
    """Get token sequence from openai-harmony."""
    conv = Conversation.from_messages(messages)
    tokens = enc.render_conversation_for_completion(conv, Role.ASSISTANT)
    return tokens


def print_token_with_channel(stream: StreamableParser, current_channel: str | None) -> str | None:
    """Print token delta and handle channel changes. Returns new channel if changed."""
    new_channel = stream.current_channel

    # Check if channel changed
    if new_channel != current_channel:
        if current_channel is not None:
            print()  # newline to end previous channel
        if new_channel:
            print(f"\n[{new_channel}] ", end="", flush=True)

    if stream.last_content_delta:
        print(stream.last_content_delta, end="", flush=True)

    return new_channel


def generate_response(
    engine: Engine,
    seq_id: int,
    all_tokens: list[int],
    cached_tokens: list[int],
    max_tokens: int = 40960,
) -> tuple[list[Message], list[int]]:
    """Generate a response given input tokens. Returns (parsed messages, updated token cache)."""
    stop_token_ids = enc.stop_tokens_for_assistant_actions()
    generated_tokens = []
    stream = StreamableParser(enc, role=Role.ASSISTANT)
    current_channel: str | None = None

    # Find common prefix and only prefill new tokens
    common_len = find_common_prefix_length(cached_tokens, all_tokens)
    new_tokens = all_tokens[common_len:]

    # prefill stage (only for new tokens)
    new_tokens_array = np.array(new_tokens, dtype=np.int32)
    prefill_logits = engine.prefill(new_tokens_array, seq_id)
    sampled_token = engine.sample(prefill_logits)
    generated_tokens.append(sampled_token)
    stream.process(sampled_token)
    current_channel = print_token_with_channel(stream, current_channel)

    # decode stage
    is_canceled = False
    while len(generated_tokens) < max_tokens:
        try:
            token_tensor = np.array([sampled_token], dtype=np.int32)
            decode_logits = engine.decode(token_tensor, seq_id)
            sampled_token = engine.sample(decode_logits)
            generated_tokens.append(sampled_token)
            stream.process(sampled_token)
            current_channel = print_token_with_channel(stream, current_channel)
            if sampled_token in stop_token_ids:
                break
        except KeyboardInterrupt:
            is_canceled = True
            break

    print()  # newline after streaming completes

    # Signal end of stream and return parsed messages
    stream.process_eos()

    # Update token cache: all input tokens + generated tokens (excluding stop token)
    updated_cache = all_tokens + generated_tokens

    return (stream.messages, updated_cache) if not is_canceled else ([], cached_tokens)


def main():
    model_path = Path.cwd() / "gpt-oss-20b" / "original"

    print("Loading model...")
    engine = Engine(model_path, target="metal")
    print("Model loaded. Type '/exit' to quit.\n")

    # Initialize conversation history with system message
    messages: list[Message] = [
        Message.from_role_and_content(
            Role.SYSTEM,
            SystemContent.new()
            .with_model_identity("You are ChatGPT, a large language model trained by OpenAI.")
            .with_reasoning_effort(ReasoningEffort.LOW)
            .with_conversation_start_date(datetime.today().strftime("%Y-%m-%d")),
        ),
    ]

    seq_id = engine.begin_sequence()
    tokens_cache: list[int] = []  # Track tokens already in KV cache

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input == "/exit":
            print("Goodbye!")
            break

        # Add user message to history
        messages.append(Message.from_role_and_content(Role.USER, user_input))

        # Get tokens for the full conversation
        all_tokens = get_input_tokens(messages)

        # Generate response (only prefill tokens not already in cache)
        response_messages, tokens_cache = generate_response(engine, seq_id, all_tokens, tokens_cache)

        # Add all assistant response messages to history
        messages.extend(response_messages)
        print()  # extra newline for spacing


if __name__ == "__main__":
    main()
