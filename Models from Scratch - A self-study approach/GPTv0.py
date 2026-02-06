"""
Esse código refere-se ao GPTv0, meu primeiro "GPT from Scratch", guiado
pelo tutorial de Andrej Karpathy. A ideia aqui é executar o código 
exatamente como no tutorial, sem nenhuma alteração. Inclusive os dados
de treino são os mesmos (Shakespeare).

A única mudança que será feita aqui é a separação do loop de treinamento
em outro código. 

1. Por questões de organização;
2. Porque eu vou reaproveitar a estrutura para treinar outras
versões do GPT (Inclusive o GPT pos minhas melhorias, ou com outros dados)
"""

# Dependencias:
import torch # Biblioteca principal para computação tensorial
import torch.nn as nn # Módulo para construir redes neurais
import torch.nn.functional as F # Funções de ativação e outras funções úteis
import requests # Biblioteca para fazer requisições HTTP

# Hyperparametros:
batch_size = 64 # Quantidade de sequencias independentes processadas em paralelo
block_size = 256 # Tamanho maximo de contexto para predição

"""
Nota sobre nomenclatura:
Talvez haja estranheza na nomenclatura "block_size" para o tamanho do
contexto. block_size aqui é equivalente a "context_size" ou "context_window",
ou seja, versa exatamente sobre a janela de contexto.

Eu estranhei, mas segundo o Claude (sonnet), "context_size" ou "context_window",
termos que estou mais acostumado, são mais comuns na literatura. Sendo block_size
um termo mais comum em tutoriais (pelo menos nos tutoriais do Karpathy).
"""

max_iters = 5000
eval_interval = 500
learning_rate = 3e-4 # 3 x 10^-4
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Usa GPU se disponível
eval_iters = 200
n_embd = 384 # Dimensão dos embeddings
n_head = 6 # Número de cabeças de atenção
n_layer = 6 # Número de camadas do Transformer
dropout = 0.2 # Taxa de dropout (desliga aleatóriamente neurônios durante o treino)

torch.manual_seed(1337) # Semente para reprodutibilidade

# ====================== Dados de Treinamento ======================
# Aquisitando dados de treinamento:
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
response = requests.get(url) # Fazendo uma requisição GET para o URL
text = response.text # Extraindo o texto da resposta

# ==================== Tokenização e vocabulário ====================
"""
Karpathy tokeniza por caracter, isso será mantido aqui. Na minha
versão, pretendo usar tokenização por sub-palavras implementando o
Byte Pair Encoding (BPE) via tiktoken (OpenAI).
"""
# Tokenizando por caracter:
chars = sorted(list(set(text))) # Lista de caracteres únicos no dataset
vocab_size = len(chars) # Tamanho do vocabulário

# Criando meapeamentos:
stoi = { ch:i for i, ch in enumerate(chars) } # Char to Index
itos = { i:ch for i, ch in enumerate(chars) } # Index to Char
encode = lambda s: [stoi[c] for c in s] # Função de codificação - transforma string em lista de inteiros
decode = lambda l: ''.join([itos[i] for i in l]) # Função de decodificação - transforma lista de inteiros em string

"""
A estratégia de tokenização também não leva em consideração caracteres
fora do conjunto de treinamento. Obviamente, a maior probabilidade é que,
dado a estratégia de tokenização, para o alfabeto ocidental, esse
problema não aconteça. Todavia, é algo a se considerar em versões
futuras.
"""

# Dividindo dados em treino e validação:
data = torch.tensor(encode(text), dtype=torch.long) # torch.long é um tipo de dado (dtype) do PyTorch que representa inteiros de 64 bits.
n = int(0.9*len(data)) # 90% dos dados para treino
train_data = data[:n] # Dados de treino
val_data = data[n:] # Dados de validação

"""
Aqui, Karpathy está considerando validação equivalente a teste.
O ideal é ter os dados de validação separados dos dados de teste,
porque validação e teste têm propósitos diferentes. 

Os dados de validação são usados para ajustar hiperparâmetros e 
avaliar o desempenho do modelo durante o treinamento, enquanto os 
dados de teste são usados apenas uma única vez no final para uma 
avaliação final objetiva do modelo.

Vou manter essa abordagem simplificada, mas em versões futuras vou
melhorar essa parte.
"""

# data loading
def get_batch(split):
    """
    Função para gerar pequenos batchs (pacotes)
    de dados de inputs x e targets y.
    """
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,)) # Índices aleatórios
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, eval_iters): # Acrescentei esses parâmetros nessa função
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
# ====================== Implementação do GPTv0 ======================
class Head(nn.Module):
    """ Uma única cabeça de auto-atenção."""

    # Método construtor
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Estrutura de mascamento para atenção causal
        self.register_buffer( # register_buffer registra um tensor que não é um parâmetro treinável
            "tril",
            torch.tril( # Tril cria uma matriz triangular inferior
                torch.ones(block_size, block_size)
                )
        )

        # Dropout para regularização
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """ Forward pass da cabeça de atenção."""
        # tamanhod do input (batch, time-step, channels)
        # tamanho da saída (batch, time-step, head size)

        # Dimensões do input
        B,T,C = x.shape

        # Calculando as matrizes de representação
        k = self.key(x) # (batch, time-step, head size)
        q = self.query(x) # (batch, time-step, head size)
        v = self.value(x) # (batch, time-step, head size)

        # Computa os scores de atenção ("afinidade")
        wei = q @ k.transpose(-2,-1) # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei / (k.shape[-1] ** 0.5)  # Escalando os scores de atenção
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei,dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # agregação ponderada dos valores
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
    
class MultiHeadAttention(nn.Module):
    """Generalização do mecânismo de auto-atenção para múltiplas cabeças"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        """Forward pass da multi-cabeça de atenção"""

        # Concatena as saídas de cada cabeça de atenção
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    """Rede neural feedforward simples"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    """Bloco Transformer: comunicação seguida de computação"""

    def __init__(self, n_embd, n_head):
        """
        Método construtor

        n_embd: dimensão dos embeddings
        n_head: número de cabeças de atenção        
        """
        super().__init__()
        head_size = n_embd // n_head # head_size diferente definido nos hyperparametros
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """Forward pass do bloco Transformer"""
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class GPT(nn.Module):
    """Modelo GPTv0 Completo"""

    # Método construtor
    def __init__(self):
        super().__init__()

        # Camada de embeddings:
        """
        Cada token lê diretamente os logits do próximo token
        a partir de uma tabela de consulta.
        """
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd) # Normalização final
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # auto-inicialização dos pesos do modelo
        """
        Melhoria na inicialização, não abordada no vídeo original
        do GPT, mas está melhoria está no código extraído do
        repositório do Karpathy.
        """
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Inicialização dos pesos do modelo.
        """

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """Forward pass do modelo GPTv0 completo"""

        B, T = idx.shape

        # idx e targets são ambos tensores (B,T) de inteiros.
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        
        # Calculando os logits
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """
        Gera novos tokens a partir de um indice inicial idx
        """

        for _ in range(idx, max_new_tokens):
            # cortar idx para os últimos tokens de block_size
            idx_cond = idx[:, -block_size:]

            # recebendo previsões
            logits, loss = self(idx_cond)

            # focando apenas no últime time step
            logits = logits[:, -1, :] # (B, C)

            # Aplicando softmax para obter probabilidades
            probs = F.softmax(logits, dim=-1) # (B, C)

            # amostras da distribuição
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # Acrescentar índice amostrado à sequência em execução
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx
