import torch
import torch.nn.functional as F

def generate(
    model,
    prompt_tokens: torch.LongTensor,   # shape: (seq_len,)
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int = None,
    device: str = 'cuda',
) -> torch.LongTensor:
    model.eval()
    # Move prompt to the same device as model
    prompt_tokens = prompt_tokens.to(device)
    generated = prompt_tokens.clone().unsqueeze(0)  # Add batch dim: (1, seq_len)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get logits for the current sequence
            logits = model(generated)  # shape: (1, current_len, vocab_size)
            # Only need the last token's logits
            next_logits = logits[0, -1, :]  # shape: (vocab_size,)

            # Temperature scaling
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Convert to probabilities
            probs = F.softmax(next_logits, dim=-1)  # shape: (vocab_size,)

            # Top-p (nucleus) sampling
            if top_p < 1.0:
                # Sort probabilities descending
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                # Compute cumulative probabilities
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                # Find indices where cumulative sum exceeds top_p
                mask = cum_probs > top_p
                # Shift mask to keep the first token that exceeds top_p
                mask = torch.cat([torch.zeros(1, device=device), mask[:-1]], dim=-1)
                # Zero out probabilities below threshold
                sorted_probs = sorted_probs * (~mask).float()
                # Renormalize the remaining probabilities
                sorted_probs = sorted_probs / sorted_probs.sum()
                # Resample: we need to sample from the reduced set
                # We'll sample from sorted indices
                next_token_idx = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_indices[next_token_idx]
            else:
                # Standard sampling
                next_token = torch.multinomial(probs, num_samples=1)

            # Append to the sequence
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

            # Stop if EOS token generated
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

    # Return as 1D tensor on CPU
    return generated.squeeze(0).cpu()