import regex as re
from collections import Counter, defaultdict
import os
import json
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

input_path="./data/TinyStoriesV2-GPT4-train.txt"
vocab_size=10000
special_tokens=["<|endoftext|>"]

def train_bpe(input_path: str,vocab_size: int ,special_tokens: list[str]):
    vocab = {i:bytes([i]) for i in range(256)}
    merges = []

    # How many times we need to merge the bytes pairs
    times= vocab_size - 256 - len(special_tokens)

    # Read the input text file
    with open(input_path, "r",encoding="utf-8") as f:
        text = f.read()
    
    special_regex = "|".join([re.escape(token) for token in special_tokens])
    parts=re.split(f"({special_regex})", text)
    segments = [p for p in parts if p if p not in special_tokens]
    print("segment done")

    freq_dict = Counter()
    for segment in segments:
        words = re.findall(PAT,segment)
        for word in words:
            freq_dict[tuple(bytes([b]) for b in word.encode("utf-8"))] += 1
    
    # We need to record the frequency of each pre-tokenized token and set a new list to store the unique tokens
    words_list=[]
    freq_list=[]
    for word,freq in freq_dict.items():
        words_list.append(list(word))
        freq_list.append(freq)

    stats = defaultdict(int)
    memory = defaultdict(set)
    
    for n,each in enumerate(words_list):
        freq=freq_list[n]
        l=len(each)
        for i in range(l-1):
            pair = (each[i],each[i+1])
            stats[pair] += freq
            memory[pair].add(n)

    print("Start Merge")

    for _ in range(times):
        if not stats:
            break
        best_pair = max(stats.items(), key=lambda x: (x[1],x[0]))[0]
        if stats[best_pair] < 1:
            break
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        relevant_words = list(memory[best_pair])
        for n in relevant_words:
            word = words_list[n]
            freq = freq_list[n]
            new_word = []
            i = 0
            while i < len(word)-1:
                if (word[i],word[i+1])==best_pair:
                    if i > 0:
                        prev_pair = (word[i-1],word[i])
                        stats[prev_pair] -= freq
                        if stats[prev_pair] <= 0:
                            del stats[prev_pair]
                    if i < len(word)-2:
                        next_pair = (word[i+1],word[i+2])
                        stats[next_pair] -= freq
                        if stats[next_pair] <= 0:
                            del stats[next_pair]
                    word[i] = new_token
                    del word[i+1]
                    if i>0:
                        prev_pair = (word[i-1],word[i])
                        stats[prev_pair] += freq
                        memory[prev_pair].add(n)
                    if i < len(word)-1:
                        next_pair = (word[i],word[i+1])
                        stats[next_pair] += freq
                        memory[next_pair].add(n)
                else:
                    i+=1
        print(f"Merge {len(merges)}: {best_pair[0]} + {best_pair[1]} -> {new_token}, frequency: {stats.get(best_pair, 0)}")
        if best_pair in stats:
            del stats[best_pair]
        if best_pair in memory:
            del memory[best_pair]   
    
    for pair in merges:
        new_id=len(vocab)
        vocab[new_id]=pair[0]+pair[1]
    for special_token in special_tokens:
        new_id=len(vocab)
        vocab[new_id]=special_token.encode("utf-8")

    return vocab,merges

def save_tokenizer(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], output_path: str):
    byte_encoder=bytes_to_unicode()
    json_vocab={
        k:"".join(byte_encoder[b] for b in v) for k,v in vocab.items()
    }
    with open(os.path.join(output_path,"vocab.json"),"w",encoding="utf-8") as f:
        json.dump(json_vocab,f,ensure_ascii=False,indent=4)

    with open(os.path.join(output_path,"merges.txt"),"w",encoding="utf-8") as f:
        for p1,p2 in merges:
            s1 = "".join(byte_encoder[b] for b in p1)
            s2 = "".join(byte_encoder[b] for b in p2)
            f.write(f"{s1} {s2}\n")

def bytes_to_unicode():
    bs = list(range(ord("!"),ord("~")+1))+list(range(ord("¡"),ord("¬")+1))+list(range(ord("®"),ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n+=1
    cs = [chr(n) for n in cs]
    return dict(zip(bs,cs))
            
    
def main():
    print(f"Start training BPE tokenizer with vocab size {vocab_size} and special tokens {special_tokens}")
    vocab,merges = train_bpe(input_path,vocab_size,special_tokens)
    output_path="./data"
    save_tokenizer(vocab,merges,output_path)

if __name__ == "__main__":
    main()