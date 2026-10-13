"""
This file is part of FLORA, an unsupervised system for automatic knowledge graph (KG) alignment. 
The file is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0) by Yiwen Peng, Thomas Bonald, Fabian Suchanek and Lingyun Huang.

Description: Literal embedding computation utilities for precomputing reusable string embeddings.
This module can be run independently from FLORA's main loop. Because literal embedding computation benefits from GPU acceleration, embeddings
can be precomputed on a GPU-enabled machine, saved to disk, and later reused by CPU-only FLORA runs.
"""

import os
import pickle
import argparse
import gc
import unicodedata
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from Prefixes import *
import literal_base
import Announce

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # safe


# Load pre-trained models for strings
DEFAULT_EMBEDDING_MODEL = 'Lihuchen/pearl_small' # 'sentence-transformers/LaBSE'
model_name = None
Str_model = None
Str_tokenizer = None
# # Multilingual BERT
# Str_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/LaBSE')
# Str_model = AutoModel.from_pretrained('sentence-transformers/LaBSE')
# Monolingual BERT
# Str_tokenizer = AutoTokenizer.from_pretrained('Lihuchen/pearl_small')
# Str_model = AutoModel.from_pretrained('Lihuchen/pearl_small')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _is_latin_text(text):
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    latin_letters = 0
    for char in letters:
        try:
            if "LATIN" in unicodedata.name(char):
                latin_letters += 1
        except ValueError:
            pass
    return latin_letters / len(letters) >= 0.8


def _normalize_for_embedding(text):
    """Normalize text for embedding."""
    return unicodedata.normalize("NFKC", text)

COMMON_NAME_PUNCTUATION = set(" .,'’`-_/&+:()[]")

def _has_suspicious_unicode(text, embedding_model=None):
    """Reject strings with suspicious Unicode characters that embedding models often collapse."""
    strict_latin = (embedding_model or DEFAULT_EMBEDDING_MODEL) == "Lihuchen/pearl_small"
    for char in text:
        if char in COMMON_NAME_PUNCTUATION:
            continue

        codepoint = ord(char)
        category = unicodedata.category(char)
        if category[0] in {"C", "S"}:  # Control or Symbol
            return True
        try:
            char_name = unicodedata.name(char)
        except ValueError:
            char_name = ""
        if any(token in char_name for token in ("REVERSED", "TURNED", "INVERTED", "SIDEWAYS")):
            return True
        if 0x02B0 <= codepoint <= 0x02FF:  # Spacing Modifier Letters
            return True
        if 0xA700 <= codepoint <= 0xA71F:  # Modifier tone letters
            return True
        if strict_latin and 0x0250 <= codepoint <= 0x02AF:  # IPA Extensions
            return True
        if strict_latin and 0x1D00 <= codepoint <= 0x1DBF:  # Phonetic Extensions
            return True
    return False

def is_embedding_literal_candidate(term, embedding_model=None):
    """Return whether a string literal should be embedded."""
    if len(term) <= 1:
        return False
    if not literal_base.is_human_readable(term):
        return False
    if _has_suspicious_unicode(term, embedding_model):
        return False

    if (embedding_model or DEFAULT_EMBEDDING_MODEL) == "Lihuchen/pearl_small":
        return _is_latin_text(term)
    return True

def load_string_model(embedding_model=None):
    global model_name, Str_model, Str_tokenizer
    selected_model = embedding_model or DEFAULT_EMBEDDING_MODEL
    if Str_model is not None and Str_tokenizer is not None and model_name == selected_model:
        return Str_model, Str_tokenizer

    model_name = selected_model
    Str_model = AutoModel.from_pretrained(model_name)
    Str_tokenizer = AutoTokenizer.from_pretrained(model_name)
    Str_model.to(device) # gpu
    Str_model.eval()
    Announce.doing(f"Pretrained Model {model_name} loaded successfully...")
    Announce.done()
    return Str_model, Str_tokenizer


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def encode_text(model, input_texts):
    if Str_tokenizer is None:
        load_string_model()
    # Tokenize the input texts
    batch_dict = Str_tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings

def embedding_strings(kb, batch_size=64, embedding_model=None, emb_file=None):
    model, _ = load_string_model(embedding_model)
    literal2id = {} # {literal:id}
    literals = []
    cnt = 0
    literal_objects = literal_base.iter_literal_objects(kb)

    for object in literal_objects:
        if literal_base.isLiteral(object):
            term, _, _, type = literal_base.splitLiteral(object)
            if type == 'xsd:string' and is_embedding_literal_candidate(term, embedding_model):
                if term in literal2id:
                    continue
                literal2id[term] = cnt
                literals.append(term)
                cnt += 1
    # Check correctness
    for ltr in literals:
        id = literal2id[ltr]
        assert literals[id] == ltr
    
    if not literals:
        hidden_size = getattr(model.config, "hidden_size", 0)
        empty_emb = np.empty((0, hidden_size), dtype=np.float32)
        if emb_file:
            np.save(emb_file, empty_emb)
            return {"id": literal2id, "emb_path": os.path.basename(emb_file)}
        return {"id": literal2id, "emb": empty_emb}

    embedding_matrix = None
    for i in tqdm(range(0, len(literals), batch_size),
                  desc="        Computing embeddings"):
        batch_literals = literals[i:i + batch_size]
        batch_inputs = [_normalize_for_embedding(literal) for literal in batch_literals]
        with torch.no_grad():
            batch_embeddings = encode_text(model, batch_inputs).cpu().numpy().astype(np.float32, copy=False)

        # Preallocate once and fill by slice.
        if embedding_matrix is None:
            shape = (len(literals), batch_embeddings.shape[1])
            # Use memory-mapped file to save memory for large embeddings.
            if emb_file:
                embedding_matrix = np.lib.format.open_memmap(emb_file, mode="w+", dtype=np.float32, shape=shape)
            else:
                embedding_matrix = np.empty(shape, dtype=np.float32)
        embedding_matrix[i:i + len(batch_literals)] = batch_embeddings
        del batch_embeddings

    if emb_file:
        embedding_matrix.flush()
        return {"id": literal2id, "emb_path": os.path.basename(emb_file)}
    return {"id": literal2id, "emb": embedding_matrix}

# Memory-efficient approach. It's useful for large knowledge graphs where loading the entire graph may not be feasible due to memory constraints.
def embedding_strings_from_ttl(path, batch_size=64, embedding_model=None, emb_file=None, fast_line_parser=False):
    """Compute embeddings by streaming TTL literals without loading the full KG."""
    model, _ = load_string_model(embedding_model)
    literal2id = {}
    literals = []

    for obj in tqdm(literal_base.iter_ttl_object_literals(path, fast_line_parser=fast_line_parser),
                    desc=f"        Scanning literals from {os.path.basename(path)}"):
        if not literal_base.isLiteral(obj):
            continue
        term, _, _, type = literal_base.splitLiteral(obj)
        if type == 'xsd:string' and is_embedding_literal_candidate(term, embedding_model):
            if term in literal2id:
                continue
            literal2id[term] = len(literals)
            literals.append(term)

    if not literals:
        hidden_size = getattr(model.config, "hidden_size", 0)
        empty_emb = np.empty((0, hidden_size), dtype=np.float32)
        if emb_file:
            np.save(emb_file, empty_emb)
            return {"id": literal2id, "emb_path": os.path.basename(emb_file)}
        return {"id": literal2id, "emb": empty_emb}

    embedding_matrix = None
    for i in tqdm(range(0, len(literals), batch_size), 
                  desc="        Computing embeddings"):
        batch_literals = literals[i:i + batch_size]
        batch_inputs = [_normalize_for_embedding(literal) for literal in batch_literals]
        with torch.no_grad():
            batch_embeddings = encode_text(model, batch_inputs).cpu().numpy().astype(np.float32, copy=False)

        if embedding_matrix is None:
            shape = (len(literals), batch_embeddings.shape[1])
            if emb_file:
                embedding_matrix = np.lib.format.open_memmap(emb_file, mode="w+", dtype=np.float32, shape=shape)
            else:
                embedding_matrix = np.empty(shape, dtype=np.float32)
        embedding_matrix[i:i + len(batch_literals)] = batch_embeddings
        del batch_embeddings

    if emb_file:
        embedding_matrix.flush()
        return {"id": literal2id, "emb_path": os.path.basename(emb_file)}
    return {"id": literal2id, "emb": embedding_matrix}

def compute_literal_embeddings(kb1, kb2, emb_path, batch_size=128, embedding_model=None):
    if not os.path.exists(emb_path):
        os.makedirs(emb_path)

    kb1_emb = embedding_strings(kb1, batch_size, embedding_model, emb_file=os.path.join(emb_path, "kb1.npy"))
    with open(os.path.join(emb_path, "kb1.pkl"), "wb") as f: # save the literal embeddings for kb1
        pickle.dump(kb1_emb, f, protocol=pickle.HIGHEST_PROTOCOL)
    del kb1_emb # free memory
    gc.collect()

    kb2_emb = embedding_strings(kb2, batch_size, embedding_model, emb_file=os.path.join(emb_path, "kb2.npy"))
    with open(os.path.join(emb_path, "kb2.pkl"), "wb") as f: # save the literal embeddings for kb2
        pickle.dump(kb2_emb, f, protocol=pickle.HIGHEST_PROTOCOL)

# Memory-efficient approach. It's useful for large knowledge graphs where loading the entire graph may not be feasible due to memory constraints.
def compute_literal_embeddings_streaming(kg1, kg2, emb_path, batch_size=128, embedding_model=None, fast_line_parser=False):
    if not os.path.exists(emb_path):
        os.makedirs(emb_path)

    kb1_emb = embedding_strings_from_ttl(
        kg1,
        batch_size,
        embedding_model,
        emb_file=os.path.join(emb_path, "kb1.npy"),
        fast_line_parser=fast_line_parser,
    )
    with open(os.path.join(emb_path, "kb1.pkl"), "wb") as f: # save the literal embeddings for kb1
        pickle.dump(kb1_emb, f, protocol=pickle.HIGHEST_PROTOCOL)
    del kb1_emb # free memory
    gc.collect()

    kb2_emb = embedding_strings_from_ttl(
        kg2,
        batch_size,
        embedding_model,
        emb_file=os.path.join(emb_path, "kb2.npy"),
        fast_line_parser=fast_line_parser,
    )
    with open(os.path.join(emb_path, "kb2.pkl"), "wb") as f: # save the literal embeddings for kb2
        pickle.dump(kb2_emb, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_params():
    parser = argparse.ArgumentParser(description="Pre-compute FLORA literal embeddings for two Turtle KGs.")
    parser.add_argument("kg1", type=str, metavar='PATH', help="Path to KG1 Turtle file, e.g., ../data/my_dataset/kg1.ttl")
    parser.add_argument("kg2", type=str, metavar='PATH', help="Path to KG2 Turtle file, e.g., ../data/my_dataset/kg2.ttl")
    parser.add_argument("emb_path", type=str, metavar='DIR', help="Output folder for kb1.pkl and kb2.pkl, e.g., ../data/emb/my_dataset/")
    parser.add_argument("--embedding_model", type=str, metavar='MODEL', default=DEFAULT_EMBEDDING_MODEL,
        help="HuggingFace model used to encode string literals. Default: Lihuchen/pearl_small. For multilingual embeddings, use sentence-transformers/LaBSE.")
    parser.add_argument("--batch_size", type=int, metavar='INT', default=128, help="Literal encoding batch size; requires positive INT")
    parser.add_argument("--literal_parser", choices=["fast", "turtle"], metavar='{fast,turtle}', default="turtle",
        help="TTL literal parser used for streaming extraction: turtle uses FLORA's Turtle parser (default); fast uses a faster scanner for one-triple-per-line files",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error('--batch_size must be positive')
    paths = vars(args)
    for key in ('kg1', 'kg2', 'emb_path'):
        if not paths[key].strip():
            parser.error('%s must not be empty' % key)
        paths[key] = os.path.abspath(os.path.expanduser(paths[key]))
    for path in (args.kg1, args.kg2):
        if not os.path.isfile(path):
            parser.error('Input file does not exist: %s' % path)
    return args


if __name__ == '__main__':
    args = get_params()
    compute_literal_embeddings_streaming(
        args.kg1,
        args.kg2,
        args.emb_path,
        batch_size=args.batch_size,
        embedding_model=args.embedding_model,
        fast_line_parser=args.literal_parser == "fast",
    )