#paramètres 
import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size=32#nombre de séquences traitées en parallèle
block_size=128  #longueur de chaque séquence
max_iters=5000 #nombre d'itérations
eval_interval=500 #intervalle d'évaluation de la loss
learning_rate=3e-4 #taux d'apprentissage

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

eval_iters=200 #nombre d'itérations pour l'évaluation de la loss
n_embed=384 #dimension des embeddings
dropout=0.2 #taux de dropout
n_layers=6#nombre de blocs empilés
n_heads=6 #nombre de têtes d'attention




class TextDataset:
    """gestion de la base de données"""
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()

        # Encodage: représentation sous forme de vecteur d'entier codé lettre par lettre 
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)

        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        self.encoded_text = torch.tensor(self.encode(self.text), dtype=torch.long)

        # Séparation en trainig data set et validation data set
        n = int(0.9 * len(self.encoded_text))
        self.train_data = self.encoded_text[:n]
        self.val_data = self.encoded_text[n:]

    def encode(self, x):
        return [self.stoi[i] for i in x]

    def decode(self, x):
        return "".join([self.itos[i] for i in x])

    # On généralise l'exemple ci-dessus à tout le texte avec la notion de batch
    def get_batch(self, selection):
        data = self.train_data if selection == 'train' else self.val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y



class Architecture(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        #On configure une table d'embedding qui va mapper chaque token de l'entrée à un vecteur de la taille du vocabulaire dont la valeur sera utilisée pour prédire le token suivant
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.positional_embedding_table = nn.Embedding(block_size, n_embed) #On encode aussi la position de chaque token dans la séquence
        #self.sa_head = MultiHeadAttention(4,n_embed//4) #Module de self-attention avec 4 têtes et une dimension par tête de n_embed/4 pour avoir une sortie de taille n_embed
        #self.ffwd = feedForward(n_embed) #Module de feedforward
        
        #On utilise block pour empiler plusieurs blocs d'attention et de feedforward
        # self.blocks = nn.Sequential(
        #     Block(n_embed,n_heads=4),
        #     Block(n_embed,n_heads=4),
        #     Block(n_embed,n_heads=4),
        #     nn.LayerNorm(n_embed) #normalisation de couche finale
        #     )
        
        self.layernorm = nn.LayerNorm(n_embed)
        
        self.blocks = nn.Sequential(*[Block(n_embed,n_heads=n_heads) for _ in range(n_layers)])
        
    def forward(self, idx, targets=None):
        #Idx est un tenseur d'indices de tokens de forme (batch_size, block_size)
        #On retourne les logits de forme (batch_size, block_size, vocab_size)
        B,T = idx.shape
        
        tok_emb = self.token_embedding_table(idx) # dimensions (B, T, n_embed)
        pos_emb = self.positional_embedding_table(torch.arange(T, device=device)) # dimensions (T, n_embed)
        x= tok_emb + pos_emb # dimensions (B, T, n_embed)
        #x=self.sa_head(x) # dimensions (B, T, n_embed) on plug le module de self-attention
        #x=self.ffwd(x) # dimensions (B, T, n_embed) on plug le module feedforward (self attention et l'aquisition de data, et le feedforward permet de laisser le temps de "comprendre les données")
        x=self.blocks(x) # dimensions (B, T, n_embed) on plug plusieurs blocs d'attention et de feedforward empilés
        x=self.layernorm(x) #normalisation de couche finale
        logits = self.lm_head(x) # dimensions (B, T, vocab_size)

        #Chaque position dans la séquence d'entrée a maintenant un vecteur de logits qui prédit le token suivant
        if targets is None: #Si les targets ne sont pas fournies, on ne calcule pas la loss afin de pouvoir faire de la génération de texte
            loss = None
        else:
            #Calcul de la loss si les targets sont fournies
            B,T,C = logits.shape #Batch size, Time steps (ou block size), Channels (ou vocab size)

            logits = logits.view(B*T, C) #On aplati les dimensions batch et time pour calculer la loss plus facilement
            targets = targets.view(B*T)  #On fait de même pour les targets
            #Mesure de Loss

            loss = F.cross_entropy(logits, targets) #cross_entropy est la negative log likelihood loss

        return logits, loss

    #Generation de texte
    def generate(self, idx, max_new_tokens):
        #idx est un tenseur de forme (batch_size, sequence_length) qui est le contexte initial
        for _ in range(max_new_tokens):
            
            idx_trcat = idx[:, -block_size:]  #On tronque à la taille maximale du block size car on a codé la position jusqu'à block_size
            
            logits, loss = self(idx_trcat)  
            #On s'intéresse seulement au dernier pas de temps
            logits = logits[:, -1, :]  #On prend les logits du dernier token de la séquence

            probs = F.softmax(logits, dim=-1)  #On convertit les logits en probabilités avec softmax

            #On échantillonne le prochain token à partir de la distribution de probabilité

            next_token = torch.multinomial(probs, num_samples=1)  #On échantillonne un token

            #On ajoute le token échantillonné à la séquence d'entrée

            idx = torch.cat((idx, next_token), dim=1)  #On concatène le nouveau token à la séquence existante
        return idx
    
    
class Head(nn.Module):
    """Module d'attention"""
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size

        self.clef = nn.Linear(n_embed, head_size, bias=False)
        self.requette = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B,T,C = x.shape
        k = self.clef(x)
        q = self.requette(x)

        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5 #(B,T,head_size) @ (B, head_size,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #(B,T,T)
        wei = F.softmax(wei, dim=-1) #(B,T,T)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v #(B,T,T) @ (B,T,head_size) -> (B,T,head_size)
        return out
        
        
        
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads*head_size, n_embed)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        sortie = torch.cat([h(x) for h in self.heads], dim=-1)
        sortie = self.dropout(self.proj(sortie))  #Desactive aléatoirement des neurones pour éviter l'overfitting
        return sortie



class feedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed,4* n_embed), #Expansion de la dimension comme dans l'article
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed ),#Projection back to n_embed
            nn.Dropout(dropout))
        #Juste un NN linéaire avec une activation ReLU
        
    def forward(self, x):
        return self.net(x)


## On souhaite empiler plusieurs blocs d'attention et de feedforward
class Block(nn.Module):
    """
    Combinaison d'un module de self-attention et d'un module feedforward
    """
    def __init__(self, n_embed,n_heads):
        super().__init__()
        head_size = n_embed // n_heads #On calcule la taille de chaque tête
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = feedForward(n_embed)
        #ajout du layer norm
        self.ln1 = nn.LayerNorm(n_embed) #normalisation de couche (moyenne 0 et variance 1) avec des paramètre gamma et beta apprenables
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # différence avec l'article : on applique le layer norm avant la self-attention et le feedforward
        x = x+self.sa(self.ln1(x)) #Residual connection
        x = x+self.ffwd(self.ln2(x)) #Residual connection
        #empiler self-attention et feedforward
        return x


class BatchNorm1d(nn.Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_var', torch.ones(dim))

    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
        else:
            x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        out = self.gamma * x_hat + self.beta
        return out

class GPTTrainer:
    """
    Gestionnaire global encapsulant le modèle, l'optimiseur et la boucle d'apprentissage.
    """
    def __init__(self, dataset_path='VER_9.txt'):
        """Initialise le modèle et l'optimiseur avec les hyperparamètres globaux."""
        self.dataset = TextDataset(dataset_path)
        self.model = Architechture(vocab_size=self.dataset.vocab_size)
        self.model = self.model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    #évaluation de la loss avec une moyenn sur eval_iters itérations pour avoir moins de bruit
    @torch.no_grad() #Tout ce qui se passe n'est pas mémorisé pour le calcul du gradient
    def estimate_loss(self):
        """Évalue et retourne la loss moyenne du modèle sur les datasets train et val."""
        out = {}
        self.model.eval() #mode évaluation
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = self.dataset.get_batch(split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train() #remettre le modèle en mode entrainement
        return out

    def train(self):
        """Exécute la boucle d'entraînement principale du modèle."""
        for steps in range(max_iters):
            
            if steps % eval_interval == 0:
                losses = self.estimate_loss()
                print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            xb, yb = self.dataset.get_batch('train')    
            logits, loss = self.model(xb, yb)

            self.optimizer.zero_grad(set_to_none=True) #on remet les gradients à zéro avant la backward pass

            loss.backward() #calcul des gradients

            self.optimizer.step()#mise à jour des poids

    def generate(self):
        """Génère du texte via le modèle entraîné."""
        #texte généré après entrainement
        idx = torch.zeros((1, 1), dtype=device == 'cuda' and torch.long or torch.long, device=device)
        print(self.dataset.decode(self.model.generate(idx, max_new_tokens=1000)[0].tolist()))

    def save(self, path):
        """Sauvegarde les poids du modèle."""
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """Charge les poids d'un modèle."""
        self.model.load_state_dict(torch.load(path))

if __name__ == "__main__":
    trainer = GPTTrainer(dataset_path='VER_9.txt')
    trainer.train()
    trainer.save('weights.pth')
    trainer.generate()
