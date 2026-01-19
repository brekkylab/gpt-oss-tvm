from pathlib import Path

import numpy as np
from openai_harmony import (
    Conversation,
    HarmonyEncodingName,
    Message,
    Role,
    StreamableParser,
    SystemContent,
    load_harmony_encoding,
)

from engine import Engine

enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def get_input_tokens():
    """get token sequence from openai-harmony or something"""
    conv = Conversation.from_messages(
        [
            Message.from_role_and_content(
                Role.SYSTEM,
                SystemContent.new().with_model_identity("You are ChatGPT, a large language model trained by OpenAI."),
            ),
            Message.from_role_and_content(Role.USER, "What is 2 + 2?"),
        ]
    )
    tokens = enc.render_conversation_for_completion(conv, Role.ASSISTANT)
    tokens = np.array(tokens).astype("int32")

    return tokens


def decode_tokens(tokens: list[int]):
    stream = StreamableParser(enc, role=Role.ASSISTANT)

    for token in tokens:
        stream.process(token)
        print("--------------------------------")
        print("current_role", stream.current_role)
        print("current_channel", stream.current_channel)
        print("last_content_delta", stream.last_content_delta)
        print("current_content_type", stream.current_content_type)
        print("current_recipient", stream.current_recipient)
        print("current_content", stream.current_content)


def main(max_token_number: int = 64):
    model_path = Path.cwd() / "gpt-oss-20b" / "original"

    engine = Engine(model_path, dump_metal_path="generated_kernel.metal")

    seq_id = engine.begin_sequence()

    stop_token_ids = enc.stop_tokens_for_assistant_actions()
    generated_tokens = []

    # prefill stage
    input_tokens = get_input_tokens()
    prefill_logits = engine.prefill(input_tokens, seq_id)
    sampled_token = engine.sample(prefill_logits)
    print(f"token: {sampled_token}, decoded: {enc.decode([sampled_token])}")
    generated_tokens.append(sampled_token)

    # decode stage
    # token_tensor = np.array([input_tokens[-1]], dtype=np.int32)
    # engine.decode(token_tensor, seq_id)

    while len(generated_tokens) < max_token_number:
        token_tensor = np.array([sampled_token], dtype=np.int32)
        decode_logits = engine.decode(token_tensor, seq_id)
        sampled_token = engine.sample(decode_logits)
        print(f"token: {sampled_token}, decoded: {enc.decode([sampled_token])}")
        generated_tokens.append(sampled_token)
        if sampled_token in stop_token_ids:
            print("Got a stop token.")
            break

    # return decode_tokens(generated_tokens)


if __name__ == "__main__":
    print(main())
