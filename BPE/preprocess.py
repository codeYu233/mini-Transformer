import os
import json
import numpy as np
from typing import List, Tuple
from BPE_Tokenizer import BPETokenizer

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def load_trained_tokenizer(vocab_json_path: str, merges_txt_path: str, special_tokens: List[str]) -> BPETokenizer:
    print(f"Loading tokenizer from {vocab_json_path}")
    byte_encoder = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}
    with open(vocab_json_path, "r", encoding="utf-8") as f:
        vocab_raw=json.load(f)
        vocab ={
            int(k): bytes([byte_decoder[b] for b in v]) for k, v in vocab_raw.items()
        }

    merges=[]
    with open(merges_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip('\n')
            if not line: continue
            parts=line.split(' ')
            if len(parts) == 2:
                merges.append((bytes([byte_decoder[c] for c in parts[0]]), bytes([byte_decoder[c] for c in parts[1]])))
    
    print(f"Vocabulary size: {len(vocab)}, Merges size: {len(merges)}")
    return BPETokenizer(vocab, merges, special_tokens)

def process_corpus(input_path: str, output_path: str, tokenizer: BPETokenizer,chunck_size_mb:int = 50):
    def file_chunck_generator(file_path: str, chunk_size: int):
        with open(file_path, "r", encoding="utf-8") as f:
            while True:
                chunck=f.read(chunk_size)
                if not chunck:
                    break
                yield chunck
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file {input_path} does not exist.")
    
    chunck_size = 1024*1024*chunck_size_mb

    if os.path.exists(output_path):
        os.remove(output_path)
    
    print(f"Processing corpus from {input_path} and saving to {output_path}")

    chunks = file_chunck_generator(input_path, chunck_size)
    token_stream=tokenizer.encode_iterable(chunks)

    total_tokens=0
    writer_batch_size=1_000_000
    token_buffer=[]
    with open(output_path, "ab") as f_out:
        for token in token_stream:
            token_buffer.append(token)
            if len(token_buffer) >= writer_batch_size:
                np_ids = np.array(token_buffer, dtype=np.uint16)
                f_out.write(np_ids.tobytes())
                total_tokens += len(token_buffer)
                token_buffer.clear()

        if token_buffer:
            np_ids = np.array(token_buffer, dtype=np.uint16)
            f_out.write(np_ids.tobytes())
            total_tokens += len(token_buffer)
            token_buffer.clear()
    
    print(f"Finished processing. Total tokens written: {total_tokens}. Output saved to {output_path}")

def main():
    BASE_DIR = "./data"
    input_path = "./data/TinyStoriesV2-GPT4-valid.txt"
    output_path = "./data/TinyStoriesV2-GPT4-valid-tokenized.bin"

    vocab_json=os.path.join(BASE_DIR, "vocab.json")
    merges_txt=os.path.join(BASE_DIR, "merges.txt")
    special_tokens = ["<|endoftext|>"]
    tokenizer=load_trained_tokenizer(vocab_json, merges_txt, special_tokens)
    process_corpus(input_path, output_path, tokenizer)

if __name__ == "__main__":
    main()