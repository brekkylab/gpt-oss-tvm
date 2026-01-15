from pathlib import Path

import numpy as np

from engine import Engine


def get_input_tokens():
    """get token sequence from openai-harmony or something"""

    # temporary input sequence: [1, 1, 1, ...]
    return np.ones((200,)).astype("int32")


def main():
    SLIDING_WINDOW_SIZE = 128

    model_path = Path.cwd() / "gpt-oss-20b" / "original"

    engine = Engine(model_path)

    seq_id = engine.begin_sequence(sliding_window_size=SLIDING_WINDOW_SIZE)
    
    # first generated token
    input_tokens = get_input_tokens()
    logits = engine.prefill(input_tokens, seq_id, sample=False)
    return logits  # temporarily return logits without sampling
    # token_id = engine.prefill(input_tokens, seq_id)

    # # second generated token
    # input_tokens = np.array((token_id,)).astype("int32")
    # token_id = engine.decode(input_tokens, seq_id)

if __name__ == "__main__":
    print(main())
