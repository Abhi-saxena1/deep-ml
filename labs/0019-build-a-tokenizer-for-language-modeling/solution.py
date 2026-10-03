def train_tokenizer(names,vocab_size):
    chars = sorted(set("".join(names)))
    vocab_size= len(chars)
    stoi= {ch: i for i , ch in enumerate(chars)}
    itos ={i: ch for i, ch in enumerate(chars)}
    def encode(name):
        return[stoi[ch] for ch in name]
    def decode(tokens):
        return "".join(itos[token] for token in tokens)
    return encode ,decode 
