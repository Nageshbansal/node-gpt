# main.py

"""Training and evaluation loop for the transformer model."""

from gpt import Transformer
import torch.optim as optim
import torch
from config import MAX_ITERS, EVAL_INTERVAL, DEVICE, LEARNING_RATE, EVAL_ITERS
from data_loading import get_batch
from data_prepration import DECODE

model = Transformer()
model.to(DEVICE)


@torch.no_grad()
def estimate_loss():
    """Estimate loss on the train and validation sets."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Create a PyTorch optimizer
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for iter in range(MAX_ITERS):

    # Every once in a while, evaluate the loss on train and val sets
    if iter % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(DECODE(model.generate(context, max_new_tokens=1000)[0].tolist()))
