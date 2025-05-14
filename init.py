import re
import TurtleUtils
from tqdm import tqdm
import math
from collections import defaultdict
import time
import pickle

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
    return decoded_string


def splitLiteral(term):
    """ Returns String value, int value, language, and datatype of a term (or None, None, None, None). No good backslash handling """

    literal, _, lang, _, datatype = re.match(literalRegex, term).groups()
    # Dates
    if datatype in ['xsd:date', 'xsd:gYear', 'xsd:gYearMonth', 'xsd:datetime', 'xsd:gMonthDay']:
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
    
    matchNumber=re.fullmatch(numberRegex, literal)
    # Check if the string is a number type, e.g., "818/762-1221"
    if matchNumber:
        Value = numeric_normalization(literal)
        return (literal, 'normalized_'+Value, lang, datatype)
    # Strings: lowecasing, order-agnostic, decode unicode
    # Pre-processing for string literals
    de_literal = decode_unicode(literal)
    return (de_literal, None, lang, 'xsd:string')

def reorder_string_with_brackets(input):
    bracket_content = re.findall(r'\(.*?\)', input)
    content_without_brackets = re.sub(r'\(.*?\)', '', input).split()
    return (' '.join(set(content_without_brackets)) + ' ' + ' '.join(bracket_content)).strip()


def numeric_normalization(term):
    term = term.strip('"')
    # Normalize the input string by removing all non-digit characters, except the sign
    # sign = term[0] if term.startswith('-') or term.startswith('+') else None
    normalized = re.sub(r'[^0-9]', '', term)
    return normalized

def is_human_readable(txt):
    non_alpha_ratio = sum(not c.isalpha() for c in txt) / len(txt) 
    # not a web link
    if txt.startswith('http:'):
        return False
    return non_alpha_ratio < 0.5


# Score literals
def scoreLiterals(literal1, literal2):
    """ Evaluates the similarity of two literals """
    split1=splitLiteral(literal1)
    split2=splitLiteral(literal2)
    # # String initialization
    # if split1[3] is None or split2[3] is None:
    #     raise ValueError('Datatype must be specified')  
    if split1[1] is not None or split2[1] is not None:  
        if split1[1] is not None and split2[1] is not None:
            # numeric handling
            if isinstance(split1[1], (int, float)) and isinstance(split2[1], (int, float)):
                # e.g., avoid code version 1.0 /= 1 interger
                if split1[3] and split2[3]:
                    # both have datatypes
                    return 1.0 if math.isclose(split1[1], split2[1]) else 0.0
                else:
                    return 1.0 if split1[0]==split2[0] else 0.0 # e.g., IDs, code versions
            return 1.0 if split1[1]==split2[1] and len(split1[1]) > 0 and len(split2[1]) > 0 else 0.0 # xsd:normalizedString
        return 0.0 # not the same datatype, e.g., int VS gYear
    return 1.0 if split1[0]==split2[0] else 0.0


# Get buckets based on types
def getLiteralBuckets(kb):
    quantityBucket = defaultdict(list) # e.g., integers, floats
    digitBucket = defaultdict(list) # e.g., IDs, code versions, identifier
    strBucket = defaultdict(list) # e.g., strings
    dateBucket = defaultdict(list) # e.g., dates
    for object1 in kb.objects():
        if isLiteral(object1):
            literal, numValue, lang, datatype = splitLiteral(object1)
            # dates handling
            if datatype in ['xsd:date', 'xsd:gYear', 'xsd:gYearMonth', 'xsd:datetime', 'xsd:gMonthDay']:
                dateBucket[literal].append(object1)
                continue
            # numeric handling
            if numValue is not None:
                if isinstance(numValue, (int, float)):
                    if datatype is not None: # strict for the specified the datatype
                        quantityBucket[numValue].append(object1)
                        continue # TBD: may delete
                        # numberBucket[numValue].append(object1)
                    # e.g., "0" vs "0"^^xsd:decimal is different
                    # serve it as a string, e.g., code version 1.0, 01
                    digitBucket[literal].append(object1)
                    continue
                # xsd:normalizedString for phone numbers, post codes, etc.
                if len(numValue) > 0:
                    strBucket[numValue].append(object1)
                continue
            # string handling
            # if is_human_readable(literal):
            strBucket[literal].append(object1)
            strBucket[literal.lower()].append(object1) # add lowercase situations
            
    return quantityBucket, digitBucket, strBucket, dateBucket


# Compare literals in same type buckets
def compareLiterals(sameAsScores, bucket1, bucket2, datatype=None):
    if datatype is None:
        raise ValueError('Datatype must be specified')
    # strings, dates
    if datatype == 'string':
        pass
    elif datatype == 'quantity':
        for key in bucket1:
            for key2 in bucket2:
                if math.isclose(key, key2):
                    for object1 in bucket1[key]:
                        if object1 not in sameAsScores:
                            sameAsScores[object1]={}
                        for object2 in bucket2[key2]:
                            sameAsScores[object1][object2] = 1.0
    elif datatype == 'date':
        for key1 in bucket1:
            if key1 in bucket2:
                for object1 in bucket1[key1]:
                    if object1 not in sameAsScores:
                        sameAsScores[object1]={}
                    for object2 in bucket2[key1]:
                        sameAsScores[object1][object2] = 1.0
                continue
            last_dash = key1.rfind('-')
            while last_dash != -1:
                key1_ = key1[:last_dash]
                if key1_ in bucket2:
                    for object1 in bucket1[key1]:
                        if object1 not in sameAsScores:
                            sameAsScores[object1]={}
                        for object2 in bucket2[key1_]:
                            sameAsScores[object1][object2] = (key1_.count('-')+1) / 3
                    break
                last_dash = key1.rfind('-', 0, last_dash)       
    else: # digits, e.g., IDs, code versions, identifier
        for key in bucket1:
            if key in bucket2 and len(key.strip('"')) > 0:
                for object1 in bucket1[key]:
                    sameAsScores[object1]={}
                    for object2 in bucket2[key]:
                        sameAsScores[object1][object2] = 1.0


def mapLiterals(kb1, kb2, path_emb1, path_emb2, threshold=0.5):
    mapScores = {}
    quantityBucket1, digitBucket1, strBucket1, dateBucket1 = getLiteralBuckets(kb1)
    quantityBucket2, digitBucket2, strBucket2, dateBucket2 = getLiteralBuckets(kb2)
    # Direction kb1 -> kb2
    # Compare strings
    # compareLiterals(mapScores, strBucket1, strBucket2, 'string')
    # compareLiterals(mapScores, strBucket2, strBucket1, 'string') # bidirectional
    # Dates
    compareLiterals(mapScores, dateBucket1, dateBucket2, 'date')
    # Compare numbers
    compareLiterals(mapScores, quantityBucket1, quantityBucket2, 'quantity')
    compareLiterals(mapScores, digitBucket1, digitBucket2, 'digit')

    # Compare strings
    for key1 in strBucket1:
        if key1 in strBucket2 and len(key1.strip('"')) > 0:
            for object1 in strBucket1[key1]:
                if object1 not in mapScores:
                    mapScores[object1]={}
                for object2 in strBucket2[key1]:
                    mapScores[object1][object2] = 1.0
    literal2id_kb1, embedding_matrix_kb1 = load_emb(path_emb1)
    literal2id_kb2, embedding_matrix_kb2 = load_emb(path_emb2)
    id2literal_kb2 = {v: k for k, v in literal2id_kb2.items()}
    id2literal_kb1 = {v: k for k, v in literal2id_kb1.items()}
    similarity_mat = embedding_matrix_kb1 @ embedding_matrix_kb2.T
    # print("Start calculating similarity matrix...")
    for key1 in strBucket1:
        if key1 in strBucket2 and len(key1.strip('"')) > 0: 
            continue
        if key1 in literal2id_kb1:
            max_indexs = similarity_mat[literal2id_kb1[key1], :].argsort()[::-1]
            for max_index in max_indexs[:1]:
                # recheck max score for strings
                if id2literal_kb2[max_index] not in strBucket2:
                    continue
                if similarity_mat[literal2id_kb1[key1], max_index] < threshold:
                    continue
                for object1 in strBucket1[key1]:
                    if object1 in mapScores and \
                        round(max(mapScores.get(object1, {None:0}).values()), 2) >= 1.0:
                        continue
                    if object1 not in mapScores:
                        mapScores[object1]={}
                    for object2 in strBucket2[id2literal_kb2[max_index]]:
                        # mapScores[object1][object2] = max_score
                        mapScores[object1][object2] = similarity_mat[literal2id_kb1[key1], max_index]    
    return mapScores


def initLiteralMapScores(mapScores, sameAsScore, kb1, kb2):
    '''
    mapScores: dict -> input dict before processing
    sameAsScore: dict -> output dict after processing
    '''
    # one-to-one literal mapping constraints (when multiple have max scores, keep them all.)
    for literal1 in mapScores:
        if literal1 not in kb1.index:
            continue
        if literal1.startswith('"tt'):
            continue
        # check empty mapping
        if literal1 not in sameAsScore:
            sameAsScore[literal1] = mapScores[literal1].copy()


def jaccard_similarity(str1, str2):
    s1 = set(str1.split())
    s2 = set(str2.split())
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return intersection / union


def load_emb(path):
    with open(path, 'rb') as file:
        emb = pickle.load(file)
    literal2id = emb['id']
    embedding_matrix = emb['emb']
    return literal2id, embedding_matrix
