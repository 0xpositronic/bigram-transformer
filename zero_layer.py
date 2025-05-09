
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(17)
torch.cuda.manual_seed(17)
device = "cuda" if torch.cuda.is_available() else "cpu"

n_embd = 128
lr = 0.8
iters = 200
data_size = 1000000
"""
vocab size: 65
bigram loss: 2.4547
transformer loss: 2.5121
Average Cosine similarity: 0.9263
Average KL divergence: 0.2720
There is still room for improvement with overly parameterized models
"""

with open('input.txt', 'r') as f:
    text = f.read()[:data_size]

chars = sorted(set(text))
vocab_size = len(chars)
print(f"vocab size: {vocab_size}")

# Tokenization. just character->number no BPE
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
encoded = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)

# Calculate bigram counts (How many times each token follows another token)
bigram_counts = torch.zeros((vocab_size, vocab_size), dtype=torch.float32)
for i in range(len(encoded)-1):
    bigram_counts[encoded[i], encoded[i+1]] += 1
# Calculate P(next token | current token)
bigram_probs = (bigram_counts + 1) / (bigram_counts + 1).sum(dim=1, keepdim=True) 
# Laplace smoothing (+1) to avoid zero probabilities for unseen bigrams which would result in -inf log-likelihood.

inputs = encoded[:-1].to(device)
targets = encoded[1:].to(device)
bigram_log_probs = torch.log(bigram_probs)
bigram_loss = -bigram_log_probs[inputs.cpu(), targets.cpu()].mean() # negative log-likelihood
print(f"bigram loss: {bigram_loss.item():.4f}")


class ZeroLayerTransformer(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.we = nn.Embedding(vocab_size, n_embd)
        self.wu = nn.Linear(n_embd, vocab_size, bias=False)
    def forward(self, x, targets=None):
        x = self.we(x)
        logits = self.wu(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1)) if targets is not None else None
        return logits, loss

model = ZeroLayerTransformer(vocab_size, n_embd).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

losses = []
for _ in range(iters):
    _, loss = model(inputs.view(1, -1), targets)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(f"transformer loss: {losses[-1]:.4f}")
# plt.plot(losses) # don't look
# plt.show()


WE = model.we.weight.cpu().detach()
WU = model.wu.weight.cpu().detach()
M = WE @ WU.T

model_probs = F.softmax(M, dim=1) # This mirrors bigram_probs. P(next token | current token)
prob_similarity = F.cosine_similarity(model_probs, bigram_probs, dim=1)
print(f"Average Cosine similarity: {prob_similarity.mean().item():.4f}")

kl_div = F.kl_div(torch.log(model_probs), bigram_probs, reduction="none").sum(dim=1)
print(f"Average KL divergence: {kl_div.mean().item():.4f}")


# uncomment to plot prob dists
'''
import seaborn as sns
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(model_probs.numpy(), xticklabels=chars, yticklabels=chars, cmap="viridis")
plt.title("Model Probabilities")
plt.subplot(1, 2, 2)
sns.heatmap(bigram_probs.numpy(), xticklabels=chars, yticklabels=chars, cmap="viridis")
plt.title("Bigram Probabilities")
plt.tight_layout()
plt.show()
'''

# extra extra (calculate difference between "e" and "E" tokens)
'''
idx_e = char_to_idx['e']
idx_E = char_to_idx['E']
vec_e = model.we.weight[idx_e].cpu().detach()
vec_E = model.we.weight[idx_E].cpu().detach()
similarity = F.cosine_similarity(vec_e.unsqueeze(0), vec_E.unsqueeze(0)).item()
distance = torch.norm(vec_e - vec_E).item()
print(f"Cosine Similarity 'e' vs 'E': {similarity:.4f}")
print(f"Euclidean Distance 'e' vs 'E': {distance:.4f}")
'''