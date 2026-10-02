import argparse
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from ordered_set import OrderedSet
from collections import Counter

from model import Transformer, ModelArgs

# =========================
# Training parameters
# =========================
SEED_FILE = 'data/filtered_ipv6_32hex.txt'
MODEL_FILE = 'data/run6MoE.pth'
CANDIDATES_FILE = 'data/candidates.txt'
FINETUNE_SEED_FILE = 'data/'

MAX_LEN = 34
BATCH_SIZE = 64
EPOCH_NUM = 50
LEARNING_RATE = 5e-4
DATA_SHUFFLE = True
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# generation parameters
TEMPERATURE = 0.8
TOP_K = 16 # <=16

# =========================
# Token system
# =========================
BOS, EOS = '<bos>', '<eos>'

tokens = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f',BOS,EOS]

token_to_id = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
               'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15, BOS: 16, EOS: 17}
id_to_token = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
               10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f', 16: BOS, 17: EOS}

BOS_ID = token_to_id[BOS]
EOS_ID = token_to_id[EOS]
VOCAB_SIZE = len(tokens)


# =========================
# Token helpers
# =========================
def token_encode(tokens_list):
    """
        Convert tokens to IDs
        nibble list -> <bos>ID + nibble ID list + <eos>ID
        token --> id
    """
    ids = [BOS_ID]
    for t in tokens_list:
        ids.append(token_to_id[t])
    ids.append(EOS_ID)
    return ids


def token_decode(token_ids):
    """
    Convert IDs to tokens
    <bos>ID + nibble ID list + <eos>ID -> nibble list
    id --> token
    """
    tokens = []
    for idx in token_ids:
        if idx not in (BOS_ID, EOS_ID):
            tokens.append(id_to_token[idx])
    return tokens


# =========================
# Dataset
# =========================
class IPv6AddrSet(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_data(seed_file, batch_size):
    """ Load the IPv6 address dataset from seed file and return a DataLoader.
        token --> id
     """
    with open(seed_file, 'r') as f:
        raw = f.readlines()

    address = []

    for line in raw:
        address.append(token_encode(line.strip()))

    dataset = IPv6AddrSet(np.array(address))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=DATA_SHUFFLE
    )

    return dataloader


# =========================
# Training
# =========================
def train_model(model,
                seed_file=SEED_FILE,
                model_file=None,
                batch_size=BATCH_SIZE,
                lr=LEARNING_RATE,
                epochs=EPOCH_NUM,
                device=DEVICE):

    dataloader = load_data(seed_file, batch_size)

    # odel Loss Function and Optimizer 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    try:

        for epoch in range(1, epochs+1):

            model.train()

            total_loss = 0

            progress = tqdm(dataloader, desc='Train...')

            for step, data in enumerate(progress, start=1):

                data = data.to(device)

                # Construct the training data and target data.
                tgt = data[:, :-1]
                tgt_y = data[:, 1:]

                out = model(tgt)
                # Transformer output: (B, S, V)

                loss = criterion(
                    out.permute(0,2,1).contiguous(),
                    tgt_y.long()
                )  #out : (B, V, S) target : (B, S)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                progress.set_description(
                    f"Train... [epoch {epoch}/{epochs}, loss {(total_loss/step):.5f}]"
                )

    except KeyboardInterrupt:
        print(f"\n训练被中断！已完成 {epoch} 轮，正在保存权重...")
    
    finally:
        # Save model parameters if needed
        if model_file is not None:
            torch.save(model.state_dict(), model_file)
            print(f"模型权重已保存至 {model_file}")
        # Return the final average training loss.
    return total_loss/step


# =========================
# Fine-tuning
# =========================
def finetune_model(model,
                   model_file,
                   alive_file,
                   batch_size=BATCH_SIZE,
                   lr=LEARNING_RATE/5,
                   epochs=5,
                   device=DEVICE):
    """
    Load model weights and fine-tune on alive addresses only.
    """
    # Load existing weights
    model.load_state_dict(torch.load(model_file, weights_only=True))
    print(f"已加载模型权重：{model_file}")

    train_model(
        model=model,
        seed_file=alive_file,
        model_file=model_file,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        device=device,
    )


# =========================
# Generation utilities
# =========================

def ids_to_ipv6(addr):
    """ Transform a list of 32 nibbles into an IPv6 address in colon-separated format.
    """
    ipv6 = ''

    for i in range(len(addr)):

        ipv6 += id_to_token[addr[i]]

        if i % 4 == 3 and i < 31:
            ipv6 += ':'

    return ipv6


# =========================
# Batch generation
# =========================

def gen_addr_batch(model, top_k, temperature, head_num, head_batch, device=DEVICE):
    """ Generate one batch of IPv6 address """
    with torch.no_grad():

        # convert IPv6 address head nibble into a tensor.
        head_batch = torch.tensor(head_batch, dtype=torch.long, device=device)

        # Strip the last <eos> token.
        tgt = head_batch[:, :-1]

        i = 0

        while i < 32 - head_num:

            out = model(tgt)

            # out: (B,S,V)
            # 'out' contains the probability distribution over all tokens in the vocabulary.
            # Exclude the last two tokens, which are <bos> and <eos>.
            #[batch_size, seq_len, vocab_size] --> [batch_size, vocab_size-2]
            logits = out[:, -1, :-2]

            # Softmax temperature scaling.
            logits = logits / temperature

            # Replace all values below top_k with -∞. 
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]

            logits[indices_to_remove] = -float('Inf')

            # Apply the softmax operation so that tokens with higher probabilities are more likely to be selected.
            probs = F.softmax(logits, dim=-1)

            # Randomly select one token from the top-k based on their probabilities.
            y = torch.multinomial(probs, num_samples=1)

            # Concatenate the selected token to the previously generated result.
            tgt = torch.cat((tgt, y), dim=-1)

            i += 1

        # Remove <bos> token and return generated addresses.
        ipv6list = list(map(ids_to_ipv6, tgt[:,1:].tolist()))

        return ipv6list


# =========================
# Prefix extraction from seed file
# =========================

def extract_prefix_budgets(seed_file, budget):
    """
    从种子文件中提取64位前缀（前16个nibble），
    统计每个前缀的频率，并按比例分配生成预算。

    Args:
        seed_file (str)
        budget (int)

    Returns:
        list of (prefix_str, prefix_budget):
    """
    with open(seed_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    prefixes = [line[:16] for line in lines]

    counter = Counter(prefixes)
    total = len(prefixes)

    prefix_budgets = []
    allocated = 0
    items = counter.most_common()  

    floor_budgets = []
    for prefix, count in items:
        pb = int(count / total * budget)
        floor_budgets.append((prefix, count, pb))
        allocated += pb

    remainder = budget - allocated  

    result = []
    for idx, (prefix, count, pb) in enumerate(floor_budgets):
        if idx < remainder:
            pb += 1
        if pb > 0:
            result.append((prefix, pb))

    return result


# =========================
# Main generation function
# =========================

def generate_target(model,
                    top_k,
                    budget,
                    candidate_file,
                    seed_file=SEED_FILE,
                    temperature=TEMPERATURE,
                    batch_size=BATCH_SIZE,
                    device=DEVICE):
    """
    Generate IPv6 addresses per /64 prefix weighted by seed frequency.
    """

    # Extract prefix → budget mapping from seed file
    prefix_budgets = extract_prefix_budgets(seed_file, budget)

    model.eval()

    all_addrs = OrderedSet()

    try:
        for prefix, prefix_budget in tqdm(prefix_budgets, desc="Prefixes"):

            head = list(prefix)           # e.g. ['2','0','0','1','1','2','4','8',...]
            head_num = len(head)          # 16

            head_tokens_ids = token_encode(head)  # [BOS, n0, n1, ..., n15, EOS]
            head_batch = [head_tokens_ids for _ in range(batch_size)]

            prefix_addrs = OrderedSet()

            max_batches = (prefix_budget // batch_size) * 5  
            iters = 0

            while len(prefix_addrs) < prefix_budget and iters < max_batches:
                gen_addr = gen_addr_batch(
                    model,
                    top_k,
                    temperature,
                    head_num,
                    head_batch,
                    device
                )
                prefix_addrs.update(gen_addr)
                print(f"\r  [{prefix}] {len(prefix_addrs)}/{prefix_budget}", end='') 
                iters += 1

            for addr in list(prefix_addrs)[:prefix_budget]:
                all_addrs.add(addr)

    except KeyboardInterrupt: 
        print(f"\n中断！已生成 {len(all_addrs)} 个地址，正在保存...")
        for addr in list(prefix_addrs):
            all_addrs.add(addr)
    
    finally:
        # Take exactly prefix_budget addresses for this prefix
        addrn = [addr + "\n" for addr in list(all_addrs)]
        with open(candidate_file, 'w') as f:
            f.writelines(addrn)
        print(f"已保存 {len(all_addrs)} 个地址到 {candidate_file}")
    



    addrn = [addr + "\n" for addr in list(all_addrs)[:budget]]
    with open(candidate_file, 'w') as f:
        f.writelines(addrn)


# =========================
# Main
# =========================

if __name__ == '__main__':
    torch.set_default_dtype(torch.bfloat16)

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed_file', default=SEED_FILE)
    parser.add_argument('--model_file', default=MODEL_FILE)
    parser.add_argument('--candidate_file', default=CANDIDATES_FILE)
    parser.add_argument('--alive_file', type=str, default=FINETUNE_SEED_FILE,
                        help='path to alive addresses file for fine-tuning')

    parser.add_argument('--epochs', type=int, default=EPOCH_NUM)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--finetune_epochs', type=int, default=5,
                        help='number of epochs for fine-tuning')

    parser.add_argument('--temperature', type=float, default=TEMPERATURE)
    parser.add_argument('--top_k', type=int, default=TOP_K)

    parser.add_argument('--budget', type=int, default=500000)
    parser.add_argument('--device', default=DEVICE)

    parser.add_argument('--no_train', action='store_true', default=True, help='no train flag')
    parser.add_argument('--finetune', action='store_true', default=False,
                        help='fine-tune on alive addresses (load weights first)')


    args = parser.parse_args()

    # create transformer args
    margs = ModelArgs()

    model = Transformer(margs).to(args.device)
    model = model.to(torch.bfloat16)

    # Model training
    if args.finetune:
        finetune_model(
            model=model,
            model_file=args.model_file,
            alive_file=args.alive_file,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            epochs=args.finetune_epochs,
            device=args.device,
        )
    elif args.no_train:
        model.load_state_dict(torch.load(args.model_file, weights_only=True))  # load model parameters
        # Generate candidate address
        generate_target(
            model=model,
            temperature=args.temperature,
            top_k=args.top_k,
            budget=args.budget,
            candidate_file=args.candidate_file,
            seed_file=args.seed_file,
            batch_size=BATCH_SIZE,
            device=args.device
        )
    else:
        train_model(
            model=model,
            seed_file=args.seed_file,
            model_file=args.model_file,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            epochs=args.epochs,
            device=args.device
        )

