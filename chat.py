from datetime import datetime
from pathlib import Path

import numpy as np
from openai_harmony import (
    Conversation,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    RenderConversationConfig,
    Role,
    StreamableParser,
    SystemContent,
    load_harmony_encoding,
)

from engine import Engine

enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def find_lcp_index(input_tokens: list[int], tokens_history: list[int]) -> int:
    """Find the length of common prefix between input tokens and history."""
    min_len = min(len(input_tokens), len(tokens_history))
    for i in range(min_len):
        if input_tokens[i] != tokens_history[i]:
            return i
    return min_len


def get_input_tokens(messages: list[Message]) -> list[int]:
    """Get token sequence from openai-harmony."""
    conv = Conversation.from_messages(messages)
    tokens = enc.render_conversation_for_completion(
        conv, Role.ASSISTANT, config=RenderConversationConfig(auto_drop_analysis=False)
    )
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
    new_tokens: list[int],
    max_tokens: int = 40960,
    use_extend: bool = False,
) -> tuple[list[Message], list[int]]:
    stop_token_ids = enc.stop_tokens_for_assistant_actions()
    generated_tokens = []
    stream = StreamableParser(enc, role=Role.ASSISTANT)
    current_channel: str | None = None

    # prefill stage (only for new tokens)
    # Use extend mode to attend to both new tokens and previously cached context
    new_tokens_array = np.array(new_tokens, dtype=np.int32)
    if use_extend:
        prefill_logits = engine.extend(new_tokens_array, seq_id)
    else:
        prefill_logits = engine.prefill(new_tokens_array, seq_id)
    print(f"current seq len after prefill: {engine.get_kv_total_seq_len()}")
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

    return (stream.messages if not is_canceled else [], generated_tokens)


def main():
    model_path = Path.cwd() / "gpt-oss-20b" / "original"

    print("Loading model...")
    engine = Engine(model_path, target="metal")
    print("Model loaded. Type '/exit' to quit.\n")

    seq_id = engine.begin_sequence()

    # Initialize conversation history with system message
    messages = [
        Message.from_role_and_content(
            Role.SYSTEM,
            SystemContent.new()
            .with_model_identity("You are ChatGPT, a large language model trained by OpenAI.")
            .with_reasoning_effort(ReasoningEffort.LOW)
            .with_conversation_start_date(datetime.today().strftime("%Y-%m-%d")),
        )
    ]
    tokens_history: list[int] = []  # Track tokens already in KV cache

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
        new_message = Message.from_role_and_content(Role.USER, user_input)
        messages.append(new_message)

        # Get tokens for the full conversation
        all_tokens = get_input_tokens(messages)

        # Rewind KV cache state
        lcp_index = find_lcp_index(all_tokens, tokens_history)
        if lcp_index < len(tokens_history):
            print(f"pop {len(tokens_history) - lcp_index} tokens from KV cache")
            engine.popn(seq_id, len(tokens_history) - lcp_index)
            tokens_history = tokens_history[:lcp_index]

        new_tokens = all_tokens[lcp_index:]

        # Generate response using extend mode to attend to both new tokens and cached context
        response_messages, generated_tokens = generate_response(
            engine,
            seq_id,
            new_tokens,
            use_extend=(len(messages) > 2),  # len(messages) == 2 means it's first time prefill
        )

        # Add all assistant response messages to history
        messages.extend(response_messages)
        tokens_history.extend(new_tokens + generated_tokens)

        print()  # extra newline for spacing


if __name__ == "__main__":
    main()
