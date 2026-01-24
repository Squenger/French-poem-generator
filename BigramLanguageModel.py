#paramètres 
import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size=32 #nombre de séquences traitées en parallèle
block_size=8  #longueur de chaque séquence
max_iters=10000 #nombre d'itérations
eval_interval=500 #intervalle d'évaluation de la loss
learning_rate=1e-3 #taux d'apprentissage
device='cuda' if torch.cuda.is_available() else 'cpu' #utilisation du GPU 
eval_iters=200 #nombre d'itérations pour l'évaluation de la loss

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Encodage: représentation sous forme de vecteur d'entier codé lettre par lettre 
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda x: [stoi[i] for i in x]
decode = lambda x: "".join([itos[i] for i in x])

encoded_text = torch.tensor(encode(text), dtype=torch.long)

# Séparation en trainig data set et validation data set
n = int(0.9 * len(encoded_text))
train_data = encoded_text[:n]
val_data = encoded_text[n:]

# On généralise l'exemple ci-dessus à tout le texte avec la notion de batch
def get_batch(selection):
    data = train_data if selection == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

#évaluation de la loss avec une moyenn sur eval_iters itérations pour avoir moins de bruit
@torch.no_grad() #Tour ce qui se passe n'est pas mémorisé pour le calcul du gradient
def estimate_loss():
    out = {}
    model.eval() #mode évaluation
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() #remettre le modèle en mode entrainement
    return out

# SImple NN pour commencer : Bigram Language Model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        #On configure une table d'embedding qui va mapper chaque token de l'entrée à un vecteur de la taille du vocabulaire dont la valeur sera utilisée pour prédire le token suivant
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        #Idx est un tenseur d'indices de tokens de forme (batch_size, block_size)
        #On retourne les logits de forme (batch_size, block_size, vocab_size)
        logits = self.token_embedding_table(idx)
        
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
            logits, loss = self(idx)  
            #On s'intéresse seulement au dernier pas de temps
            logits = logits[:, -1, :]  #On prend les logits du dernier token de la séquence

            probs = F.softmax(logits, dim=-1)  #On convertit les logits en probabilités avec softmax

            #On échantillonne le prochain token à partir de la distribution de probabilité

            next_token = torch.multinomial(probs, num_samples=1)  #On échantillonne un token

            #On ajoute le token échantillonné à la séquence d'entrée

            idx = torch.cat((idx, next_token), dim=1)  #On concatène le nouveau token à la séquence existante
        return idx

model = BigramLanguageModel(vocab_size)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for steps in range(max_iters):
    
    if steps % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')    
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True) #on remet les gradients à zéro avant la backward pass

    loss.backward() #calcul des gradients

    optimizer.step()#mise à jour des poids

#texte généré après entrainement
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(idx, max_new_tokens=1000)[0].tolist()))