import sys, torch
sys.path.insert(0, '/root/CN_seg/seven/seg')
sys.path.insert(1, '/root/CN_seg/seven')
from models import MAESegmenter
import config_seg as config

model = MAESegmenter(config.MAE_CHECKPOINT).cuda()
x = torch.randn(2, 1, 256, 256).cuda()
logits = model(x)
loss = logits.mean()
loss.backward()

enc_grad = sum(p.grad.abs().mean().item() for p in model.encoder.parameters() if p.grad is not None)
dec_grad = sum(p.grad.abs().mean().item() for p in model.decoder.parameters() if p.grad is not None)
enc_no_grad = sum(1 for p in model.encoder.parameters() if p.grad is None)
print(f'Encoder grad sum: {enc_grad:.6f}')
print(f'Decoder grad sum: {dec_grad:.6f}')
print(f'Encoder params with no grad: {enc_no_grad}')
