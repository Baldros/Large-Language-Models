"""
Script de treinamento do GPT. Preferi separar
do codigo de construção do modelo para ficar
mais organizado.
"""
import torch
from GPTv0 import (
                    GPT, 
                    get_batch, 
                    device, 
                    learning_rate, 
                    max_iters, 
                    eval_interval,
                    eval_iters,
                    decode,
                    estimate_loss,
                )

llm = GPT().to(device)

# printando o número de parâmetros no modelo
print(sum(p.numel() for p in llm.parameters())/1e6, 'M parameters')

# Instanciando otimizador PyTorch AdamW
optimizer = torch.optim.AdamW(llm.parameters(),lr = learning_rate)

for iter in range(max_iters):
    # De tempos em tempos, avalie a perda nos conjuntos de treinamento e validação
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(llm, eval_iters)
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # Minibatch de dados
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = llm(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# gerando texto após o treinamento
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(llm.generate(context, max_new_tokens=500)[0].tolist()))