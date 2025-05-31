import torch

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(100, 64)
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=64, nhead=2, batch_first=True), 
            num_layers=2
        )
        self.lm_head = torch.nn.Linear(64, 100)
    
    def forward(self, idx):
        x = self.embedding(idx)
        x = self.transformer(x)
        logits = self.lm_head(x)
        return logits
    
    def generate(self, idx, max_new_tokens, **kwargs):
        # Simple dummy generation that just returns the input
        return idx

model = DummyModel()
model_args = {
    'n_layer': 2,
    'n_head': 2,
    'n_embd': 64,
    'block_size': 128,
    'vocab_size': 100,
    'dropout': 0.0,
}

checkpoint = {
    'model': model.state_dict(),
    'model_args': model_args,
    'iter_num': 0,
    'best_val_loss': 999.0,
    'config': {'dataset': 'cogprime'}
}

torch.save(checkpoint, 'nanoGPT/out-nanocog-ci/ckpt.pt')