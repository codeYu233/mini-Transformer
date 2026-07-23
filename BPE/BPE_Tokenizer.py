import regex as re
from collections import Counter, defaultdict
import os
import json
from typing import Iterable

class BPETokenizer:
    def __init__(self,vocab:dict[int,bytes],merges:list[tuple[bytes,bytes]],special_tokens:list[str]):
        self.vocab=vocab
        self.id2byte = vocab
        self.byte2id = {v:k for k,v in vocab.items()}
        self.merges={pair:i for i,pair in enumerate(merges)}
        self.special_tokens=special_tokens
        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        sorted_special = sorted(self.special_tokens,key=lambda x:len(x),reverse=True)
        special_pattern = "|".join([re.escape(token) for token in sorted_special])
        self.special_regex = re.compile(special_pattern)

    def _encode_text_segment(self,text:str)->list[int]:
        ids=[]
        pre_tokens = self.PAT.findall(text)
        for p_token in pre_tokens:
            byte_seq = [bytes([b]) for b in p_token.encode("utf-8")]
            while len(byte_seq)>1:
                best_pair=None
                min_rank=float('inf')
                for i in range(len(byte_seq)-1):
                    pair = (byte_seq[i],byte_seq[i+1])
                    if pair in self.merges:
                        rank = self.merges[pair]
                        if rank < min_rank:
                            min_rank=rank
                            best_pair=pair
                if best_pair is None:
                    break
                new_byte_seq=[]
                i=0
                while i < len(byte_seq):
                    if i < len(byte_seq)-1 and (byte_seq[i],byte_seq[i+1])==best_pair:
                        new_byte_seq.append(best_pair[0]+best_pair[1])
                        i+=2
                    else:
                        new_byte_seq.append(byte_seq[i])
                        i+=1
                byte_seq=new_byte_seq
            for seq in byte_seq:
                ids.append(self.byte2id[seq])
        return ids

    def encode(self,text:str)->list[int]:
        if not text:
            return []
        tokens=[]
        last_pos=0
        for match in self.special_regex.finditer(text):
            pre_text=text[last_pos:match.start()]
            if pre_text:
                tokens.extend(self._encode_text_segment(pre_text))
            tokens.append(self.byte2id[match.group().encode("utf-8")])
            last_pos=match.end()
        post_text=text[last_pos:]
        if post_text:
            tokens.extend(self._encode_text_segment(post_text))
        return tokens
    
    def decode(self,ids: list[int])->str:
        byte_seq=b"".join(self.id2byte[i] for i in ids)
        return byte_seq.decode("utf-8",errors="replace")

    def encode_iterable(self,iterable:Iterable[str])->Iterable[list[int]]:
        for chunk in iterable:
            yield from self.encode(chunk)