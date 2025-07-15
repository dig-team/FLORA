import re
import pickle
import sys
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from Prefixes import *
import TurtleUtils

# Load pre-trained models for strings
# Multilingual BERT
Str_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/LaBSE')
Str_model = AutoModel.from_pretrained('sentence-transformers/LaBSE')
# Monolingual BERT
# Str_tokenizer = AutoTokenizer.from_pretrained('Lihuchen/pearl_small')
# Str_model = AutoModel.from_pretrained('Lihuchen/pearl_small')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Str_model.to(device) # gpu
print("Pretrained Models loaded successfully...")


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def encode_text(model, input_texts):
    # Tokenize the input texts
    batch_dict = Str_tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings


def string_similarity(input_texts):
    '''
    input_texts: list of strings [source, target1, target2, ...]
    '''
    embeddings = encode_text(Str_model, input_texts)
    scores = (embeddings[:1] @ embeddings[1:].T) # no * 100
    return scores.tolist()


def is_readable(txt):
    if len(txt.split()) > 1:
        return True
    non_alpha_ratio = sum(not c.isalpha() for c in txt) / len(txt) 
    # not a web link
    if txt.startswith('http'):
        return False
    return non_alpha_ratio < 0.5



# Regex for literals
literalRegex=re.compile('"([^"]*)"(@([a-z-]+))?(\\^\\^(.*))?')

# Regex for int values
intRegex=re.compile('^"?[+-]?[0-9]+"?$')

# Regex for float values
floatRegex = re.compile('^"?([+-])?([0-9.]+)"?$')
sciFloatRegex = re.compile('^"?([+-])?([0-9.]+[Ee][+-]?[0-9]+)"?$')

# Regex for numbers: post code, phone number, etc.
numberRegex = re.compile('[\d\W]+')
identifierRegex = re.compile('([a-zA-Z]+)(\d+)') # TBD if needed


def isLiteral(term):
    return re.match(literalRegex,term) or re.match(floatRegex,term)

def unicode(txt):
    # e.g., "Coamo,_Puerto_Rico" -> "Coamo_u002C_Puerto_Rico"
    encoded_str = re.sub(r'[^a-zA-Z0-9_]', lambda x: "_u{:04X}_".format(ord(x.group())), txt)
    return encoded_str

def decode_unicode(encoded_str):
    decoded_string = re.sub(r'_u([0-9A-F]{4})_', lambda x: chr(int(x.group(1), 16)), encoded_str)
    if '\\u' in decoded_string:
        try:
            s = decoded_string.encode('utf-8').decode('unicode_escape')
            return s.encode('utf-16', 'surrogatepass').decode('utf-16')
        except Exception:
            return decoded_string
    else:
        return decoded_string


def splitLiteral(term):
    """ Returns String value, int value, language, and datatype of a term (or None, None, None, None). No good backslash handling """
    
    literal, _, lang, _, datatype = re.match(literalRegex, term).groups()
    # Dates
    if datatype in ['xsd:date', 'xsd:gYear', 'xsd:gYearMonth', 'xsd:datetime']:
        return (literal, None, lang, datatype)
    
    # Numbers
    floatmatch = re.match(floatRegex, literal)
    scifloatmatch = re.match(sciFloatRegex, literal)
    if (floatmatch or scifloatmatch) and lang is None:
        try:
            # some identifiers are integers
            Value=int(literal.strip('"'))
            if len(str(Value)) != len(literal.strip('"')):
                # e.g., "06" /= 6, "+3" /= 3
                return (literal, None, lang, datatype) # datatype 'none'
            return (literal, Value, lang, datatype)
        except:
            try:
                # e.g., code version 1.0 /= 1
                Value=float(literal.strip('"'))
                return (literal, Value, lang, datatype)
            except ValueError:
                # e.g. "23.78.9" version
                return (literal, None, lang, datatype)
    
    # Cases for normalized numbers, e.g., phone numbers, post codes
    # Would put it to the string bucket
    matchNumber=re.fullmatch(numberRegex, literal)
    # Check if the string is a number type, e.g., "818/762-1221"
    if matchNumber:
        Value = numeric_normalization(literal)
        if len(Value) >= 10: # e.g., phone numbers, not: ', , , ,'; "1.2"@fr
            return (literal, 'normalized_'+Value, lang, datatype)
    
    # Strings: lowecasing, order-agnostic, decode unicode
    # Pre-processing for string literals
    de_literal = decode_unicode(literal)
    return (de_literal, None, lang, 'xsd:string')


def embedding_strings(kb, batch_size=64):
    literal2id = {} # {literal:id}
    literals = []
    cnt = 0
    for object in kb.objects():
        if isLiteral(object):
            term, _, _, type = splitLiteral(object)
            if len(term) <= 1:
                print("Empty/Short literal: ", term, object)
                continue
            if not is_readable(term) and type == 'xsd:string':
                print("Unreadable literal: ", term, object)
                continue
            if type == 'xsd:string' and is_readable(term):
                if term in literal2id:
                    continue
                literal2id[term] = cnt
                literals.append(term)
                cnt += 1
    # Check correctness
    for ltr in literals:
        id = literal2id[ltr]
        assert literals[id] == ltr
    
    # literals
    # Compute embeddings in batches
    embeddings_list = []
    for i in tqdm(range(0, len(literals), batch_size),
                  desc="Computing embeddings"):
        batch_literals = literals[i:i + batch_size]
        with torch.no_grad():
            embeddings = encode_text(Str_model, batch_literals).cpu()
            embeddings_list.append(embeddings)
    
    # Stack embeddings to form the matrix
    embedding_matrix = torch.cat(embeddings_list, dim=0)
    # convert to numpy
    embedding_matrix = embedding_matrix.numpy()
    return {"id": literal2id, "emb": embedding_matrix}
    

def numeric_normalization(term):
    term = term.strip('"')
    # Normalize the input string by removing all non-digit characters, except the sign
    # sign = term[0] if term.startswith('-') or term.startswith('+') else None
    normalized = re.sub(r'[^0-9]', '', term)
    return normalized

def jaccard_similarity(str1, str2):
    s1 = set(str1.split())
    s2 = set(str2.split())
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return intersection / union



if __name__ == '__main__':

    if len(sys.argv)<2:
        print("python literals.py <dataset path> <emb_path>")
        exit()

    # print("Loading first KB...")
    # kb1=TurtleUtils.graphFromTurtleFile(sys.argv[1])
    # kb1predicates=kb1.predicates()

    # print("Loading second KB...")
    # kb2=TurtleUtils.graphFromTurtleFile(sys.argv[2])
    # kb2predicates=kb2.predicates()

    # kb1, kb2, gt = TurtleUtils.load_openea('/home/infres/ypeng-21/work/OM/Paris2/github/Paris2/camera-ready/FLORA/data/D_W_15K_V2')
    kb1, kb2 = TurtleUtils.load_dbp15k(sys.argv[1], attr=True, name=True)

    emb_path = sys.argv[2]
    batch_size = 128
    kb1_emb = embedding_strings(kb1, batch_size)
    kb2_emb = embedding_strings(kb2, batch_size)

    # save embeddings
    with open(emb_path+"kb1_emb.pkl", "wb") as f:
        pickle.dump(kb1_emb, f)
    with open(emb_path+"kb2_emb.pkl", "wb") as f:
        pickle.dump(kb2_emb, f)