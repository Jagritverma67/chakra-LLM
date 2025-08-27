# ------------------------------------------------------
# test_chakra.py
# Simple demo for Chakra LLM (forward + generation)
# ------------------------------------------------------

from app import ChakraConfig, ChakraTransformer, ChakraTokenizer
import torch

def main():
    # Load tokenizer
    tokenizer = ChakraTokenizer("gpt2")

    # Build model config
    cfg = ChakraConfig(
        vocab_size=tokenizer.vocab_size,
        max_tokens=64,
        embed_dim=128,
        layers=2,
        heads=4,
        dropout=0.1
    )
    model = ChakraTransformer(cfg)

    # Input text
    text = "hello world, this is Chakra testing!"
    tokens = tokenizer.encode(text)
    x = torch.tensor([tokens], dtype=torch.long)

    # Forward pass
    logits, loss = model(x, x)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item()}")

    # Generate continuation
    out = model.generate(x, max_new_tokens=20, temperature=1.0, top_k=50)
    print("Generated:", tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
