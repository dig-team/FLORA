"""
This file is part of FLORA, an unsupervised system for automatic knowledge graph (KG) alignment. 
The file is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0) by Yiwen Peng, Thomas Bonald, Fabian Suchanek and Lingyun Huang.

Description: Graph data structures, compact graph representations, RDF/Turtle parsers and dataset loaders.
"""

import os
import codecs
import re
import sys
import shutil
import pickle
from io import StringIO
import Prefixes
from collections import OrderedDict, defaultdict
from functools import reduce
from tqdm import tqdm
import numpy as np

##########################################################################
#             Predicate URI canonicalization
##########################################################################

def _build_full_uri_to_compact_predicate():
    """Build a deterministic full-URI to compact-prefix predicate map."""
    preferred_prefixes = OrderedDict()
    for prefix_map in (Prefixes.prefixes, Prefixes.prefixes_dbp):
        for prefix, uri in prefix_map.items():
            preferred_prefixes.setdefault(prefix, uri)

    full_to_compact = {}
    for prefix, uri in preferred_prefixes.items():
        full_to_compact[uri] = prefix + ':'
    return full_to_compact

FULL_URI_TO_COMPACT_PREFIX = _build_full_uri_to_compact_predicate()

def canonicalizePredicate(predicate):
    """Canonicalize predicate URI. This only rewrites full URI to their known compact prefix form,
    e.g. ``<http://www.w3.org/2000/01/rdf-schema#label>`` to ``rdfs:label``.
    """
    inverse = isInverse(predicate) if predicate else False
    base = predicate[:-1] if inverse else predicate
    if base.startswith('<') and base.endswith('>'):
        base = base[1:-1]
    for uri, compact_prefix in FULL_URI_TO_COMPACT_PREFIX.items():
        if base.startswith(uri) and len(base) > len(uri):
            return compact_prefix + base[len(uri):] + ('-' if inverse else '')
    return predicate

##########################################################################
#             Parsing Turtle
##########################################################################

def printError(*args, **kwargs):
    """ Prints an error to StdErr """
    print(*args, file=sys.stderr, **kwargs)

def termsAndSeparators(generator):
    """ Iterator over the terms of char reader """
    pushBack=None
    while True:
        # Scroll to next term
        while True:
            char=pushBack if pushBack else next(generator, None)
            pushBack=None
            if not char:
                # end of file
                yield None
                return
            elif char=='@':
                # @base and @prefix
                for term in termsAndSeparators(generator):
                    if not term:
                        printError("Unexpected end of file in directive")
                        return
                    if term=='.':
                        break
            elif char=='#':
                # comments
                while char and char!='\n':
                    char=next(generator, None)
            elif char.isspace():
                # whitespace
                pass
            else:
                break

        # Strings
        if char=='"':
            secondChar=next(generator, None)
            thirdChar=next(generator, None)
            if secondChar=='"' and thirdChar=='"':
                # long string quote
                literal=""
                while True:
                    char=next(generator, None)
                    if char:
                        literal=literal+char
                    else:
                        printError("Unexpected end of file in literal",literal)
                        literal=literal+'"""'
                        break
                    if literal.endswith('"""'):
                        break
                literal=literal[:-3]
                char=None
            else:
                # Short string quote
                if secondChar=='"':
                    literal=''
                    char=thirdChar
                elif thirdChar=='"' and secondChar!='\\':
                    literal=secondChar
                    char=None
                else:
                    literal=[secondChar,thirdChar]
                    if thirdChar=='\\' and secondChar!='\\':
                        literal+=next(generator, ' ')
                    while True:
                        char=next(generator, None)
                        if not char:
                            printError("Unexpected end of file in literal",literal)
                            break
                        elif char=='\\':
                            literal+=char
                            literal+=next(generator, ' ')
                            continue
                        elif char=='"':
                            break
                        literal+=char
                    char=None
                    literal="".join(literal)
            # Make all literals simple literals without line breaks and quotes
            literal=literal.replace('\n','\\n').replace('\t','\\t').replace('\r','').replace('\\"',"'").replace("\\u0022","'")
            if not char:
                char=next(generator, None)
            if char=='^':
                # Datatypes
                next(generator, None)
                datatype=''
                while True:
                    char=next(generator, None)
                    if not char:
                        printError("Unexpected end of file in datatype of",literal)
                        break
                    if len(datatype)>0 and datatype[0]!='<' and char!=':' and (char<'A' or char>'z') and char!='/' and (char!='2' and char!='3') \
                        and char != 'ó' and char != 'ł': # exceptions: m^2, /km2, ó, polishZłoty
                        pushBack=char
                        break
                    datatype=datatype+char
                    if datatype.startswith('<') and datatype.endswith('>'):
                        break
                if not datatype or len(datatype)<3:
                    printError("Invalid literal datatype:", datatype)
                yield('"'+literal+'"^^'+datatype)
            elif char=='@':
                # Languages
                language=""
                while True:
                    char=next(generator, None)
                    if not char:
                        printError("Unexpected end of file in language of",literal)
                        break
                    if (char>='A' and char<='Z') or (char>='a' and char<='z') or (char>='0' and char<='9') or char=='-':
                        language+=char
                        continue
                    pushBack=char
                    break
                if not language or len(language)>20 or len(language)<2 or ('-' in language and len(language[language.index('-'):])>9):
                    if TEST:
                        printError("Invalid literal language:", language)
                    yield('"'+literal+'"')
                else:
                    yield('"'+literal+'"@'+language)
            else:
                pushBack=char
                yield('"'+literal+'"')
        elif char=='<':
            # URIs
            uri=[]
            while char!='>':
                uri+=char
                char=next(generator, None)
                if not char:
                    printError("Unexpected end of file in URL",uri)
                    break
            uri+='>'
            yield "".join(uri)
        elif char in ['.',',',';','[',']','(',')']:
            # Separators
            yield char
        else:
            # Local names
            iri=[]
            while not char.isspace() and char not in ['.',',',';','[',']','"',"'",'^','@','(',')']:
                iri+=char
                char=next(generator, None)
                if not char:
                    printError("Unexpected end of file in IRI",iri)
                    break
            pushBack=char
            yield "".join(iri)

# Counts blank nodes to give a unique name to each of them
blankNodeCounter=0

def blankNodeName(subject, predicate=None):
    """ Generates a legible name for a blank node in the YS namespace """
    global blankNodeCounter
    if ':' in subject:
        lastIndex=len(subject) - subject[::-1].index(':') - 1
        subject=subject[lastIndex+1:]+"_"
    elif predicate:
        subject=""
    if predicate and ':' in predicate:
        lastIndex=len(predicate) - predicate[::-1].index(':') - 1
        predicate=predicate[lastIndex+1:]
    else:
        predicate=""
    blankNodeCounter+=1
    return "ys:"+subject+predicate+"_"+str(blankNodeCounter)

def triplesFromTerms(generator, predicates=None, givenSubject=None):
    """ Iterator over the triples of a term generator """
    while True:
        term=next(generator, None)
        if not term or term==']':
            return
        if term=='.' or (term==';' and givenSubject):
            continue
        # If we're inside a [...]
        if givenSubject:
            subject=givenSubject
            if term!=',':
                predicate=term
        # If we're in a normal statement
        else:
            if term!=';' and term!=',':
                subject=term
            if term!=',':
                predicate=next(generator, None)
        if predicate=='a':
            predicate='rdf:type'
        # read the object
        object=next(generator, None)
        if not object:
            printError("File ended unexpectedly after", subject, predicate)
            return
        elif object in ['.',',',';']:
            printError("Unexpected",object,"after",subject,predicate)
            return
        elif object=='(':
            listNode=blankNodeName("list")
            previousListNode=None
            yield (subject, predicate, listNode)
            while True:
                term=next(generator, None)
                if not term:
                    printError("Unexpected end of file in collection (...)")
                    break
                elif term==')':
                    break
                else:
                    if previousListNode:
                        yield (previousListNode, 'rdf:rest', listNode)
                    if term=='[':
                        term=blankNodeName("element")
                        yield (listNode, 'rdf:first', term)
                        yield from triplesFromTerms(generator, predicates, givenSubject=term)
                    else:
                        yield (listNode, 'rdf:first', term)
                    previousListNode=listNode
                    listNode=blankNodeName("list")
            yield (previousListNode, 'rdf:rest', 'rdf:nil')
        elif object=='[':
            object=blankNodeName(subject, predicate)
            yield (subject, predicate, object)
            yield from triplesFromTerms(generator, predicates, givenSubject=object)
        else:
            if (not predicates) or (predicate in predicates):
                yield (subject, predicate, object)

##########################################################################
#             Reading files
##########################################################################

def byteGenerator(byteReader):
    """ Generates bytes from the reader """
    while True:
        b=byteReader.read(1)
        if b:
            yield b
        else:
            break

def charGenerator(byteGenerator):
    """ Generates chars from bytes """
    return codecs.iterdecode(byteGenerator, "utf-8")

def triplesFromTurtleFile(file, message=None, predicates=None):
    """ Iterator over the triples in a TTL file """
    if message:
        print(message+"... ",end="",flush=True)
    with open(file,"rb") as reader:
        yield from triplesFromTerms(termsAndSeparators(charGenerator(byteGenerator(reader))), predicates)
    if message:
        print("done", flush=True)

def graphFromTurtleFile(file, message=None):
    """ Returns a graph for a Turtle file """
    graph=Graph()
    for triple in triplesFromTurtleFile(file, message):
        graph.add(triple)
    return graph

##########################################################################
#             Graphs
##########################################################################

def isInverse(rel):
    """ TRUE if the relation is an inverse relation """
    return rel[-1]=='-'

def invert(rel):
    """ Returns the inverse of a relation """
    return rel[:-1] if isInverse(rel) else rel+'-'

class Graph(object):
    """ A graph of triples """
    def __init__(self, biparti=True):
        self.index = {} # {subject:{predicate:set(object)}}
        self.relindex = {} # {predicate:{subject:set(object)}}
        self.pred2num = None # predicate -> number of facts with this predicate
        return

    def add(self, triple):
        (subject, predicate, obj) = triple
        predicate = canonicalizePredicate(predicate)
        if subject not in self.index:
            self.index[subject] = {}
        m = self.index[subject]
        if predicate not in m:
            m[predicate] = set()
        m[predicate].add(obj)

        # relindex
        if predicate not in self.relindex:
            self.relindex[predicate] = {}
        m = self.relindex[predicate]
        if subject not in m:
            m[subject] = set()
        m[subject].add(obj)

        if not isInverse(predicate):
            self.add((obj, invert(predicate), subject))
        self.pred2num = None

    def remove(self, triple):
        (subject, predicate, obj) = triple
        predicate = canonicalizePredicate(predicate)
        if subject not in self.index:
            return
        m = self.index[subject]
        if predicate not in m:
            return
        m[predicate].discard(obj)
        if len(m[predicate]) == 0:
            self.index[subject].pop(predicate)
            if len(self.index[subject]) == 0:
                self.index.pop(subject)

        # relindex
        if predicate in self.relindex and subject in self.relindex[predicate]:
            self.relindex[predicate][subject].discard(obj)
            if len(self.relindex[predicate][subject]) == 0:
                self.relindex[predicate].pop(subject)
                if len(self.relindex[predicate]) == 0:
                    self.relindex.pop(predicate)

        if not isInverse(predicate):
            self.remove((obj, invert(predicate), subject))
        self.pred2num = None

    def _objects_for_subject_predicate(self, subject, predicate):
        """ Returns the set of objects for the given subject and predicate, or None if there are no such facts. """
        predicate = canonicalizePredicate(predicate) if predicate else predicate
        predicate_map = self.index.get(subject)
        if predicate_map is None:
            return None
        return predicate_map.get(predicate)

    def __contains__(self, triple):
        (subject, predicate, obj) = triple
        predicate = canonicalizePredicate(predicate)
        objects = self._objects_for_subject_predicate(subject, predicate)
        return objects is not None and obj in objects

    def has_subject(self, subject):
        return subject in self.index

    def iter_view_subjects(self):
        """ Returns an iterator over the subjects of the graph """
        return iter(self.index)

    def __iter__(self):
        for subject in self.iter_view_subjects():
            for predicate, objects in self.subject_items(subject):
                for obj in objects:
                    yield (subject, predicate, obj)

    def loadTurtleFile(self, file, message=None):
        for triple in triplesFromTurtleFile(file, message):
            self.add(triple)

    def getList(self, listStart):
        """ Returns the elements of an RDF list"""
        result=[]
        while listStart and listStart!='rdf:nil':
            result.extend(self._objects_for_subject_predicate(listStart, 'rdf:first') or [])
            rest_objects = self._objects_for_subject_predicate(listStart, 'rdf:rest')
            if not rest_objects:
                break
            listStart=list(rest_objects)[0]
        return result

    def predicates(self):
        if self.pred2num is None:
            self.numFactsWithPredicate("blah")
        return self.pred2num

    def attributes(self):
        """ Returns all the attributes of the graph """
        result=set()
        for predicate in self.predicates():
            if self.isAttribute(predicate):
                result.add(predicate)
                result.add(invert(predicate)) # add inverse
        return result

    def numFactsWithPredicate(self, predicate):
        predicate = canonicalizePredicate(predicate)
        if self.pred2num is not None:
            return self.pred2num[predicate] if predicate in self.pred2num else 0
        self.pred2num = {}
        for subject in self.index:
            for pred in self.index[subject]:
                if pred not in self.pred2num:
                    self.pred2num[pred] = 0
                self.pred2num[pred] += len(self.index[subject][pred])
        return self.pred2num[predicate] if predicate in self.pred2num else 0

    def isAttribute(self, pred):
        """ Returns TRUE if the predicate is an attribute"""
        pred = canonicalizePredicate(pred)
        predicate_map = self.relindex.get(pred, {})
        for subject in predicate_map:
            objects = predicate_map[subject]
            for obj in objects:
                if isLiteral(obj):
                    return True
        return False

    def localFunctionality(self, subjects, preds):
        if not isinstance(subjects, (list, tuple)):
            subjects = [subjects]
        if not isinstance(preds, (list, tuple)):
            preds = [preds]
        if len(subjects) != len(preds):
            raise ValueError("The input size of subjects and predicates are not equal.")
        commonObjs = None
        for i in range(len(subjects)):
            pred = canonicalizePredicate(preds[i])
            objects = self._objects_for_subject_predicate(subjects[i], pred)
            if objects is None:
                commonObjs = set()
                break
            if commonObjs is None:
                commonObjs = objects
            else:
                commonObjs = commonObjs & objects
        try:
            value = 1.0/len(commonObjs)
        except ZeroDivisionError:
            value = 0
        return value

    def objects(self, subject=None, predicate=None):
        predicate = canonicalizePredicate(predicate) if predicate else predicate
        # We create a copy here instead of using a generator
        if subject is None and predicate is None:
            result = set()
            for predicate_map in self.index.values():
                for objects in predicate_map.values():
                    result.update(objects)
            return result

        result=[]
        if subject and not self.has_subject(subject):
            return result

        if subject:
            for _, objects in self.subject_items(subject, [predicate] if predicate else None):
                result.extend(objects)
            return set(result)

        if predicate is None:
            return self.objects()

        predicate_map = self.relindex.get(predicate, {})
        for objects in predicate_map.values():
            result.extend(objects)
        return set(result)

    def subjects(self, predicate=None, object=None):
        predicate = canonicalizePredicate(predicate) if predicate else predicate
        pred = invert(predicate) if predicate else None
        return self.objects(subject=object, predicate=pred)

    def triplesWithObject(self, obj, predicates=[]):
        return self.triplesWithSubject(obj, [invert(p) for p in predicates])

    def triplesWithSubject(self, subject, predicates=[]):
        predicates = [canonicalizePredicate(predicate) for predicate in predicates] if predicates else predicates
        for predicate, objects in self.subject_items(subject, predicates if len(predicates) else None):
            for obj in objects:
                yield (subject, predicate, obj)

    def triplesWithPredicate(self, *predicates):
        predicates = tuple(canonicalizePredicate(predicate) for predicate in predicates)
        for subject in self.iter_view_subjects():
            for predicate in predicates:
                objects = self._objects_for_subject_predicate(subject, predicate)
                if not objects:
                    continue
                for obj in objects:
                    yield (subject, predicate, obj)

    def headTriplesWithPredicateList(self, predicatesWithCount):
        """ Returns the triples dictionary where the head entity
            has all predicates in the given predicates """
        result = {} # {head: pred: tail}
        predicatesWithCount = {
            canonicalizePredicate(predicate): count
            for predicate, count in predicatesWithCount.items()
        }
        heads = [set(self.subjects(predicate=predicate)) for predicate in predicatesWithCount]
        sharedHeads = reduce(lambda x, y: x & y, heads)

        for predicate in set(predicatesWithCount):
            required_count = predicatesWithCount[predicate]
            for subject in sharedHeads:
                objects = self._objects_for_subject_predicate(subject, predicate)
                if objects is None or len(objects) < required_count:
                    continue
                if subject not in result:
                    result[subject] = set()
                # To avoid high complexity for type, genre relations,
                # which have a super hugh number of subjects
                pred2count_ = defaultdict(int)
                for obj in objects:
                    result[subject].add((subject, predicate, obj))
                    pred2count_[predicate] += 1
                    if pred2count_[predicate] > 10:
                        break
                # result[subject][predicate].update(self.relindex[predicate][subject])
        return result

    def subject_items(self, subject, predicates=None):
        """ Returns an iterator over the (predicate, objects) pairs for the given subject and predicates. """
        if not self.has_subject(subject):
            return
        if predicates:
            predicates = [canonicalizePredicate(predicate) for predicate in predicates]
            for predicate in predicates:
                objects = self._objects_for_subject_predicate(subject, predicate)
                if objects:
                    yield predicate, objects
            return
        # if predicates is None, return all predicates and objects for the subject
        for predicate, objects in self.index.get(subject, {}).items():
            yield predicate, objects

    def printToWriter(self, result):
        for subject in self.iter_view_subjects():
            if subject.startswith("_:list_"):
                continue
            result.write("\n")
            result.write(subject)
            result.write(' ')
            hasPreviousPred=False
            for predicate, objects in self.subject_items(subject):
                if isInverse(predicate):
                    continue
                if hasPreviousPred:
                    result.write(' ;\n\t')
                hasPreviousPred=True
                result.write(predicate)
                result.write(' ')
                hasPrevious=False
                for obj in objects:
                    if hasPrevious:
                        result.write(', ')
                    if obj.startswith("_:list_"):
                        result.write("(")
                        result.write(" ".join(self.getList(obj)))
                        result.write(")")
                    else:
                        result.write(obj)
                    hasPrevious=True
            result.write(' .\n')

    def __str__(self):
        buffer=StringIO()
        buffer.write("# RDF Graph\n")
        self.printToWriter(buffer)
        return buffer.getvalue()

    def someSubject(self):
        for key in self.iter_view_subjects():
            return key
        return None

    def __len__(self):
        # Total number of facts
        return sum(self.predicates().values())


class CompactSymbolTable(object):
    """String-to-ID table used by CompactGraph."""

    _FILENAME = 'symbols.pkl'

    def __init__(self, entity2id=None, id2entity=None, predicate2id=None, id2predicate=None):
        self.entity2id = entity2id if entity2id is not None else {}
        self.id2entity = id2entity if id2entity is not None else []
        self.predicate2id = predicate2id if predicate2id is not None else {}
        self.id2predicate = id2predicate if id2predicate is not None else []

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as handle:
            payload = pickle.load(handle)
        if isinstance(payload, cls):
            return payload
        return cls(payload.get('entity2id'), payload.get('id2entity'), payload.get('predicate2id'), payload.get('id2predicate'))

    def save(self, path):
        with open(path, 'wb') as handle:
            pickle.dump(
                {'entity2id': self.entity2id, 'id2entity': self.id2entity, 'predicate2id': self.predicate2id, 'id2predicate': self.id2predicate},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def add_entity(self, entity):
        entity_id = self.entity2id.get(entity)
        if entity_id is None:
            entity_id = len(self.id2entity)
            self.entity2id[entity] = entity_id
            self.id2entity.append(entity)
        return entity_id

    def add_predicate(self, predicate):
        predicate_id = self.predicate2id.get(predicate)
        if predicate_id is None:
            predicate_id = len(self.id2predicate)
            self.predicate2id[predicate] = predicate_id
            self.id2predicate.append(predicate)
        return predicate_id

    def entity_count(self):
        return len(self.id2entity)

    def predicate_count(self):
        return len(self.id2predicate)


class CompactGraph(object):
    """Compact graph representation backed by mmap arrays for large KGs."""

    _COUNT_CACHE_MAXSIZE = 262144 # max size for the local functionality count cache
    _LOCAL_FUNCTIONALITY_IDS_CACHE_MAXSIZE = 262144 # max size for multi subject-predicate local functionality cache
    _OBJECT_IDS_CACHE_MAXSIZE = 65536 # max size for the subject-predicate pair to object ids cache
    _OBJECT_IDS_CACHE_MAX_OBJECTS = 4096 # max number of objects for a subject-predicate pair to be cached

    def __init__(self):
        self._mmap_dir = None
        self._name = None
        self._symbols = CompactSymbolTable() # id -> string and string -> id mappings for entities and predicates
        self._symbols_path = None

        # Viewed edges, including inverse edges, sorted by (subject, predicate, object).
        self._objects = None
        self._subject_pair_offsets = None # subject -> offset into subject-predicate pairs
        self._subject_pair_predicates = None # subject-predicate pair -> predicate id for the pair
        self._subject_pair_fact_offsets = None # subject-predicate pair -> offset into objects array for the pair's facts

        # Per-entity and per-predicate arrays for quick property checks and counts.
        self._entity_is_literal = None # entity id -> whether the entity is a literal, used for quick literal checks without decoding the id
        self._inverse_predicate_ids = None # predicate id -> inverse predicate id, used for quick inverse checks without decoding the id
        self._predicate_counts = None # predicate id -> count of facts with the predicate, used for quick predicate count checks for predicates alignment

        # Per-process hot caches
        self._local_functionality_count_cache = OrderedDict() # (subject, predicate) -> count of objects for the subject-predicate pair, used for local functionality calculation
        self._local_functionality_ids_cache = OrderedDict() # (tuple of subject ids, tuple of predicate ids) -> count of common objects for the subject-predicate pairs, used for local functionality calculation
        self._subject_predicate_object_ids_cache = OrderedDict() # (subject, predicate) -> set of object ids for the subject-predicate pair, used for local functionality calculation and (subject, predicate) -> object ids queries

    @classmethod
    def from_graph(cls, graph, mmap_dir, name):
        """Builds a CompactGraph from the given Graph, storing the internal arrays in the specified directory with the given name."""
        compact = cls()
        compact._mmap_dir = os.path.abspath(os.path.join(mmap_dir, name))
        compact._name = name
        compact._symbols_path = os.path.join(compact._mmap_dir, CompactSymbolTable._FILENAME)
        if os.path.exists(compact._mmap_dir):
            shutil.rmtree(compact._mmap_dir)
        os.makedirs(compact._mmap_dir, exist_ok=True)
        compact._build_from_graph(graph)
        compact._save_symbols()
        return compact

    def __getstate__(self):
        """Prepares the state for pickling, excluding the large mmap arrays and caches."""
        state = self.__dict__.copy()
        symbols_path = state.get('_symbols_path')
        if symbols_path and os.path.exists(symbols_path):
            state['_symbols'] = None
        # Exclude large arrays from the pickled state
        for key in (
            '_objects',
            '_subject_pair_offsets',
            '_subject_pair_predicates',
            '_subject_pair_fact_offsets',
            '_entity_is_literal',
            '_inverse_predicate_ids',
            '_predicate_counts',
        ):
            state[key] = None
        state['_local_functionality_count_cache'] = OrderedDict()
        state['_local_functionality_ids_cache'] = OrderedDict()
        state['_subject_predicate_object_ids_cache'] = OrderedDict()
        return state

    def __setstate__(self, state):
        """Restores the state from pickling, reopening the mmap arrays and reinitializing the caches."""
        self.__dict__.update(state)
        self._local_functionality_count_cache = OrderedDict()
        self._local_functionality_ids_cache = OrderedDict()
        self._subject_predicate_object_ids_cache = OrderedDict()
        self._open_arrays()

    def _symbols_file_path(self):
        if self._symbols_path is None and self._mmap_dir is not None:
            self._symbols_path = os.path.join(self._mmap_dir, CompactSymbolTable._FILENAME)
        return self._symbols_path

    def _save_symbols(self):
        path = self._symbols_file_path()
        if path is not None and self._symbols is not None:
            self._symbols.save(path)

    def _require_symbols(self):
        if self._symbols is None:
            path = self._symbols_file_path()
            if path is None:
                raise RuntimeError("CompactGraph symbol table is unavailable.")
            self._symbols = CompactSymbolTable.load(path)
        return self._symbols

    @property
    def _entity2id(self):
        return self._require_symbols().entity2id

    @property
    def _id2entity(self):
        return self._require_symbols().id2entity

    @property
    def _predicate2id(self):
        return self._require_symbols().predicate2id

    @property
    def _id2predicate(self):
        return self._require_symbols().id2predicate

    def _array_path(self, name):
        return os.path.join(self._mmap_dir, name + '.npy')

    def _save_array(self, name, data):
        path = self._array_path(name)
        arr = np.lib.format.open_memmap(path, mode='w+', dtype=data.dtype, shape=data.shape)
        arr[:] = data
        arr.flush()
        return np.load(path, mmap_mode='r')

    def _open_arrays(self):
        if self._mmap_dir is None:
            return
        self._objects = np.load(self._array_path('objects'), mmap_mode='r')
        self._subject_pair_offsets = np.load(self._array_path('subject_pair_offsets'), mmap_mode='r')
        self._subject_pair_predicates = np.load(self._array_path('subject_pair_predicates'), mmap_mode='r')
        self._subject_pair_fact_offsets = np.load(self._array_path('subject_pair_fact_offsets'), mmap_mode='r')
        self._entity_is_literal = np.load(self._array_path('entity_is_literal'), mmap_mode='r')
        self._inverse_predicate_ids = np.load(self._array_path('inverse_predicate_ids'), mmap_mode='r')
        self._predicate_counts = np.load(self._array_path('predicate_counts'), mmap_mode='r')

    def entity_id(self, entity):
        return self._entity2id.get(entity)

    def predicate_id(self, predicate):
        return self._predicate2id.get(predicate)

    def entity_for_id(self, entity_id):
        return self._id2entity[int(entity_id)]

    def predicate_for_id(self, predicate_id):
        return self._id2predicate[int(predicate_id)]

    def num_entities(self):
        return self._require_symbols().entity_count()

    def num_predicates(self):
        return self._require_symbols().predicate_count()

    def iter_view_subjects(self):
        return range(self.num_entities())

    def iter_predicate_ids(self):
        return range(self.num_predicates())

    def _inverse_predicate_id(self, predicate_id):
        """Returns the inverse predicate ID for the given predicate ID."""
        if predicate_id is None:
            return None
        predicate_id = int(predicate_id)
        if self._inverse_predicate_ids is None or predicate_id < 0 or predicate_id >= len(self._inverse_predicate_ids):
            return None
        inverse_id = int(self._inverse_predicate_ids[predicate_id])
        return inverse_id if inverse_id >= 0 else None

    def is_literal_id(self, entity_id):
        """Return whether an entity id is a literal."""
        if entity_id is None:
            return False
        entity_id = int(entity_id)
        if self._entity_is_literal is None:
            return bool(isLiteral(self._id2entity[entity_id]))
        if entity_id < 0 or entity_id >= len(self._entity_is_literal):
            return False
        return bool(self._entity_is_literal[entity_id])

    def iter_literal_object_ids(self):
        """Yield all literal objects."""
        for object_id in range(self.num_entities()):
            if self.is_literal_id(object_id):
                yield object_id

    def predicate_counts_ids(self):
        """Return a dictionary mapping predicate IDs to their counts in the graph."""
        if self._predicate_counts is None:
            if self._subject_pair_predicates is None or self._subject_pair_fact_offsets is None:
                return {}
            counts = {}
            for pair_idx, predicate_id in enumerate(self._subject_pair_predicates):
                start, end = self._pair_fact_bounds(self._subject_pair_fact_offsets, pair_idx)
                predicate_id = int(predicate_id)
                counts[predicate_id] = counts.get(predicate_id, 0) + (end - start)
            return counts
        return {predicate_id: int(count) for predicate_id, count in enumerate(self._predicate_counts) if count}

    def predicates(self): # to be compatible with the original graph
        return {self._id2predicate[predicate_id]: int(count) for predicate_id, count in self.predicate_counts_ids().items()}

    def _build_from_graph(self, graph):
        """ Builds the compact graph from the given graph. """
        triples = []
        symbols = self._require_symbols()

        # Encode entities and predicates and collect triples as (subject_id, predicate_id, object_id)
        for subject in graph.iter_view_subjects():
            subject_id = symbols.add_entity(subject)
            for predicate, objects in graph.subject_items(subject):
                predicate_id = symbols.add_predicate(predicate)
                for obj in objects:
                    object_id = symbols.add_entity(obj)
                    triples.append((subject_id, predicate_id, object_id))

        entity_count = symbols.entity_count()
        predicate_count = symbols.predicate_count()
        raw_fact_count = len(triples)

        id_dtype = np.int64 if entity_count > np.iinfo(np.int32).max else np.int32
        pred_dtype = np.int32 if predicate_count <= np.iinfo(np.int32).max else np.int64
        offset_dtype = np.int64

        # Create numpy arrays for subjects, predicates, and objects
        subjects = np.empty(raw_fact_count, dtype=id_dtype)
        predicates = np.empty(raw_fact_count, dtype=pred_dtype)
        objects = np.empty(raw_fact_count, dtype=id_dtype)
        for idx, (subject_id, predicate_id, object_id) in enumerate(triples):
            subjects[idx] = subject_id
            predicates[idx] = predicate_id
            objects[idx] = object_id
        del triples

        if raw_fact_count:
            # Sort the triples by (subject, predicate, object)
            order = np.lexsort((objects, predicates, subjects))
            subjects = subjects[order]
            predicates = predicates[order]
            objects = objects[order]
            # Remove duplicate triples
            unique = np.empty(raw_fact_count, dtype=bool)
            unique[0] = True
            unique[1:] = ((subjects[1:] != subjects[:-1]) | (predicates[1:] != predicates[:-1]) | (objects[1:] != objects[:-1]))
            subjects = subjects[unique]
            predicates = predicates[unique]
            objects = objects[unique]
        fact_count = len(subjects)

        # Compute predicate counts, entity literal flags, and inverse predicate IDs
        predicate_counts = np.bincount(predicates, minlength=predicate_count)
        entity_is_literal = np.asarray([bool(isLiteral(entity)) for entity in symbols.id2entity], dtype=bool)
        inverse_predicate_ids = np.full(predicate_count, -1, dtype=pred_dtype)
        for predicate_id, predicate in enumerate(symbols.id2predicate):
            inverse_predicate_id = symbols.predicate2id.get(invert(predicate))
            if inverse_predicate_id is not None:
                inverse_predicate_ids[predicate_id] = inverse_predicate_id

        # Build subject-predicate pair arrays
        if fact_count:
            # Identify the start of each subject-predicate pair in the sorted triples
            pair_starts = np.empty(fact_count, dtype=bool)
            pair_starts[0] = True
            pair_starts[1:] = ((subjects[1:] != subjects[:-1]) | (predicates[1:] != predicates[:-1]))
            pair_start_indices = np.flatnonzero(pair_starts).astype(offset_dtype, copy=False)
            # predicate id for each subject-predicate pair
            subject_pair_predicates = predicates[pair_start_indices]
            # offset into objects array for each subject-predicate pair
            subject_pair_fact_offsets = np.empty(len(pair_start_indices) + 1, dtype=offset_dtype)
            subject_pair_fact_offsets[:-1] = pair_start_indices
            subject_pair_fact_offsets[-1] = fact_count
            # count of subject-predicate pairs for each subject
            subject_pair_counts = np.bincount(subjects[pair_start_indices], minlength=entity_count).astype(offset_dtype, copy=False)
        else:
            subject_pair_predicates = np.empty(0, dtype=pred_dtype)
            subject_pair_fact_offsets = np.zeros(1, dtype=offset_dtype)
            subject_pair_counts = np.zeros(entity_count, dtype=offset_dtype)
        # offsets for each subject in the subject-predicate pairs
        subject_pair_offsets = np.empty(entity_count + 1, dtype=offset_dtype)
        subject_pair_offsets[0] = 0
        np.cumsum(subject_pair_counts, out=subject_pair_offsets[1:])

        # Save the shared arrays to disk and memory-map them for efficient access
        self._objects = self._save_array('objects', objects)
        self._subject_pair_offsets = self._save_array('subject_pair_offsets', subject_pair_offsets)
        self._subject_pair_predicates = self._save_array('subject_pair_predicates', subject_pair_predicates)
        self._subject_pair_fact_offsets = self._save_array('subject_pair_fact_offsets', subject_pair_fact_offsets)
        self._entity_is_literal = self._save_array('entity_is_literal', entity_is_literal)
        self._inverse_predicate_ids = self._save_array('inverse_predicate_ids', inverse_predicate_ids)
        self._predicate_counts = self._save_array('predicate_counts', predicate_counts)

    def _find_pair_index(self, pair_offsets, pair_predicates, owner_id, predicate_id):
        """ Finds the index of the pair (owner_id, predicate_id) in the pair_predicates array using binary search. Returns None if the pair is not found. """
        low = int(pair_offsets[owner_id])
        high = int(pair_offsets[owner_id + 1])
        while low < high:
            mid = (low + high) // 2
            mid_predicate = int(pair_predicates[mid])
            if mid_predicate < predicate_id:
                low = mid + 1
            elif mid_predicate > predicate_id:
                high = mid
            else:
                return mid
        return None

    def _pair_fact_bounds(self, pair_fact_offsets, pair_idx):
        """ Returns the start and end indices in the objects array for the given pair index. """
        start = int(pair_fact_offsets[pair_idx])
        end = int(pair_fact_offsets[pair_idx + 1])
        return start, end

    def clear_hot_caches(self):
        """Clear all per-process hot cache entries."""
        self._local_functionality_count_cache.clear()
        self._local_functionality_ids_cache.clear()
        self._subject_predicate_object_ids_cache.clear()

    def _lru_get(self, cache, key):
        try:
            value = cache.pop(key)
        except KeyError:
            return None
        cache[key] = value
        return value

    def _lru_put(self, cache, key, value, maxsize):
        cache[key] = value
        if len(cache) > maxsize:
            cache.popitem(last=False)

    def _subject_predicate_pair_index(self, subject_id, predicate_id):
        """Return the pair index for the given (subject_id, predicate_id)."""
        if subject_id is None or predicate_id is None:
            return None
        if subject_id < 0 or subject_id + 1 >= len(self._subject_pair_offsets):
            return None
        return self._find_pair_index(
            self._subject_pair_offsets,
            self._subject_pair_predicates,
            int(subject_id),
            int(predicate_id),
        )

    def _subject_predicate_fact_indices(self, subject_id, predicate_id):
        """Return the object-array index range for a subject-predicate pair."""
        pair_idx = self._subject_predicate_pair_index(subject_id, predicate_id)
        if pair_idx is None:
            return ()
        start, end = self._pair_fact_bounds(
            self._subject_pair_fact_offsets,
            pair_idx,
        )
        return range(start, end)

    def _object_ids_for_subject_predicate_ids(self, subject_id, predicate_id, cache_key=None):
        """Fast path to get the object IDs for a subject-predicate pair, using caches."""
        if cache_key is None:
            cache_key = (
                subject_id,
                predicate_id if predicate_id is not None else -1,
            )
        cached = self._lru_get(
            self._subject_predicate_object_ids_cache,
            cache_key,
        )
        if cached is not None:
            return cached

        object_ids = []
        if predicate_id is not None:
            for fact_idx in self._subject_predicate_fact_indices(subject_id, predicate_id):
                object_id = int(self._objects[fact_idx])
                object_ids.append(object_id)

        result = tuple(object_ids)
        if len(result) <= self._OBJECT_IDS_CACHE_MAX_OBJECTS:
            self._lru_put(
                self._subject_predicate_object_ids_cache,
                cache_key,
                result,
                self._OBJECT_IDS_CACHE_MAXSIZE,
            )
        return result


    def _subject_predicate_fact_count(self, subject_id, predicate_id):
        """Return the number of objects for the given (subject_id, predicate_id)."""
        pair_idx = self._subject_predicate_pair_index(subject_id, predicate_id)
        if pair_idx is None:
            return 0
        start, end = self._pair_fact_bounds(
            self._subject_pair_fact_offsets,
            pair_idx,
        )
        return end - start

    def _objects_for_subject_predicate_id_count(self, subject_id, predicate_id):
        """Fast path to get the number of objects for a subject-predicate pair, using caches."""
        if subject_id is None:
            return 0
        subject_id = int(subject_id)
        cache_key = (
            subject_id,
            predicate_id if predicate_id is not None else -1,
        )
        cached_count = self._lru_get(
            self._local_functionality_count_cache,
            cache_key,
        )
        if cached_count is not None:
            return cached_count

        cached_object_ids = self._lru_get(
            self._subject_predicate_object_ids_cache,
            cache_key,
        )
        if cached_object_ids is not None:
            count = len(cached_object_ids)
            self._lru_put(
                self._local_functionality_count_cache,
                cache_key,
                count,
                self._COUNT_CACHE_MAXSIZE,
            )
            return count

        count = (
            self._subject_predicate_fact_count(subject_id, predicate_id)
            if predicate_id is not None else 0
        )
        self._lru_put(
            self._local_functionality_count_cache,
            cache_key,
            count,
            self._COUNT_CACHE_MAXSIZE,
        )
        return count

    def _has_object_for_subject_predicate_id(self, subject_id, predicate_id, object_id):
        """Return whether an subject-predicate edge contains object_id."""
        if subject_id is None or predicate_id is None or object_id is None:
            return False
        pair_idx = self._subject_predicate_pair_index(int(subject_id), int(predicate_id))
        if pair_idx is None:
            return False
        start, end = self._pair_fact_bounds(self._subject_pair_fact_offsets, pair_idx)
        if start == end:
            return False
        objects = self._objects[start:end]
        index = np.searchsorted(objects, int(object_id))
        return bool(index < len(objects) and int(objects[index]) == int(object_id))

    def has_subject_id(self, subject_id):
        """Return whether the given subject ID has any associated facts."""
        if subject_id is None:
            return False
        subject_id = int(subject_id)
        if subject_id < 0 or subject_id + 1 >= len(self._subject_pair_offsets):
            return False
        return int(self._subject_pair_offsets[subject_id + 1]) > int(self._subject_pair_offsets[subject_id])

    def iterFactIds(self):
        """Yield fact triples as integer ids."""
        for subject_id in range(self.num_entities()):
            pair_start = int(self._subject_pair_offsets[subject_id])
            pair_end = int(self._subject_pair_offsets[subject_id + 1])
            for pair_idx in range(pair_start, pair_end):
                predicate_id = int(self._subject_pair_predicates[pair_idx])
                fact_start, fact_end = self._pair_fact_bounds(
                    self._subject_pair_fact_offsets,
                    pair_idx,
                )
                for fact_idx in range(fact_start, fact_end):
                    yield (subject_id, predicate_id, int(self._objects[fact_idx]))

    def localFunctionalityIds(self, subject_ids, predicate_ids):
        """Return the local functionality for the given subject and predicate IDs."""
        # local functionality for a single subject-predicate pair
        if not isinstance(subject_ids, (list, tuple)) and not isinstance(predicate_ids, (list, tuple)):
            count = self._objects_for_subject_predicate_id_count(subject_ids, predicate_ids)
            return 1.0 / count if count else 0

        if not isinstance(subject_ids, (list, tuple)):
            subject_ids = [subject_ids]
        if not isinstance(predicate_ids, (list, tuple)):
            predicate_ids = [predicate_ids]
        if len(subject_ids) != len(predicate_ids):
            raise ValueError("The input size of subjects and predicates are not equal.")
        if len(subject_ids) == 1:
            count = self._objects_for_subject_predicate_id_count(subject_ids[0], predicate_ids[0])
            return 1.0 / count if count else 0

        # local functionality for multiple subject-predicate pairs
        cache_key = (
            tuple(int(subject_id) for subject_id in subject_ids),
            tuple(int(predicate_id) for predicate_id in predicate_ids),
        )
        cached_value = self._lru_get(
            self._local_functionality_ids_cache,
            cache_key,
        )
        if cached_value is not None:
            return cached_value

        # Compute the intersection of object IDs for all subject-predicate pairs
        common_object_ids = None
        for i in range(len(subject_ids)):
            subject_id = subject_ids[i]
            predicate_id = predicate_ids[i]
            object_ids = self._object_ids_for_subject_predicate_ids(
                subject_id,
                predicate_id,
                None,
            )
            if not object_ids:
                common_object_ids = set()
                break
            if common_object_ids is None:
                common_object_ids = set(object_ids)
            else:
                common_object_ids = common_object_ids & set(object_ids)
        try:
            value = 1.0 / len(common_object_ids)
        except ZeroDivisionError:
            value = 0
        self._lru_put(
            self._local_functionality_ids_cache,
            cache_key,
            value,
            self._LOCAL_FUNCTIONALITY_IDS_CACHE_MAXSIZE,
        )
        return value

    def triplesWithSubjectIds(self, subject_id):
        """Yield all ID triples for the given subject ID."""
        if subject_id is None:
            return
        subject_id = int(subject_id)
        if not self.has_subject_id(subject_id):
            return
        # Iterate over the subject-predicate pairs for the given subject ID and yield triples
        pair_start = int(self._subject_pair_offsets[subject_id])
        pair_end = int(self._subject_pair_offsets[subject_id + 1])
        for pair_idx in range(pair_start, pair_end):
            predicate_id = int(self._subject_pair_predicates[pair_idx])
            fact_start, fact_end = self._pair_fact_bounds(self._subject_pair_fact_offsets, pair_idx)
            for fact_idx in range(fact_start, fact_end):
                yield (subject_id, predicate_id, int(self._objects[fact_idx]))

    def triplesWithSubjectIdsFiltered(self, subject_id, predicate_id_set):
        """Returns an iterator over triples with the given subject ID and a set of predicate IDs. If predicate_id_set is empty, returns no triples."""
        if not predicate_id_set:
            return
        if subject_id is None:
            return
        subject_id = int(subject_id)
        if not self.has_subject_id(subject_id):
            return
        if not isinstance(predicate_id_set, (set, frozenset)):
            predicate_id_set = set(int(predicate_id) for predicate_id in predicate_id_set)

        pair_start = int(self._subject_pair_offsets[subject_id])
        pair_end = int(self._subject_pair_offsets[subject_id + 1])
        for pair_idx in range(pair_start, pair_end):
            predicate_id = int(self._subject_pair_predicates[pair_idx])
            # only yield triples for predicates in the given set
            if predicate_id not in predicate_id_set:
                continue
            fact_start, fact_end = self._pair_fact_bounds(self._subject_pair_fact_offsets, pair_idx)
            for fact_idx in range(fact_start, fact_end):
                yield (subject_id, predicate_id, int(self._objects[fact_idx]))

    def __len__(self):
        return len(self._objects)


# Regex for literals
literalRegex=re.compile('"([^"]*)"(@([a-z-]+))?(\\^\\^(.*))?')

# Regex for int values
intRegex=re.compile('^"?[+-]?[0-9.]+"?$')

def isLiteral(term):
    return re.match(literalRegex,term) or re.match(intRegex,term)


##########################################################################
#                 Useful functions
##########################################################################


def load_openea(loc, attr=True):
    """ Loads OpenEA datasets """
    gt_pairs = []
    with open(os.path.join(loc, 'ent_links'), 'r', encoding='UTF-8') as f:
            for line in f:
                head, tail = line.strip().split('\t')
                gt_pairs.append((head, tail))

    kg1, kg2=Graph(), Graph()
    for i in tqdm(range(2), desc='   Loading OpenEA {name}...'.format(name=loc.split('/')[-2])):
         with open(os.path.join(loc,'rel_triples_{}'.format(i+1)), 'r', encoding='UTF-8') as f:
                for line in f.readlines():
                    head, rel, tail = line.strip().split('\t')
                    if i == 0:
                        kg1.add((head, rel, tail))
                    else:
                        kg2.add((head, rel, tail))
    if attr: # load attributes as well
         for i in range(2):
             with open(os.path.join(loc,'attr_triples_{}'.format(i+1)), 'r', encoding='UTF-8') as f:
                 for line in f.readlines():
                     head, attribute, literal = line.strip().split('\t')
                     # process literals
                     if literal.startswith('"'):
                        # Datatypes
                        literal = literal.replace('\n','\\n').replace('\t','\\t').replace('\r','').replace('\\"',"'").replace("\\u0022","'")
                        if '^^' in literal:
                            str_value, datatype = literal.split('^^')
                            prefixed_datatype = re.sub(r'<http://www\.w3\.org/2001/XMLSchema#([a-zA-Z0-9_-]+)>', r'xsd:\1', datatype)
                            literal = '"'+str_value[1:-1]+'"^^'+prefixed_datatype
                            if i == 0:
                                kg1.add((head, attribute, literal))
                            else:
                                kg2.add((head, attribute, literal))
                            continue
                        # other situation
                        if i == 0:
                            kg1.add((head, attribute, literal))
                        else:
                            kg2.add((head, attribute, literal))
                     else:
                        # Make all literals simple literals without line breaks and quotes
                        literal = literal.replace('\n','\\n').replace('\t','\\t').replace('\r','').replace('\\"',"'").replace("\\u0022","'")
                        if i == 0:
                            kg1.add((head, attribute, '"'+literal+'"'))
                        else:
                            kg2.add((head, attribute, '"'+literal+'"'))
    return kg1, kg2, gt_pairs


def load_dbp15k(loc, trans=False, attr=True, name=True):
    """ Loads DBP15K datasets """
    kg1, kg2 = Graph(), Graph()
    # id2ent1, id2ent2 = {}, {} # for seed pairs if needed
    for i in tqdm(range(2), desc='   Loading DBP15K {name}...'.format(name=loc.split('/')[-2])):
        id2rel, id2ent = {}, {}
        with open(os.path.join(loc, 'rel_ids_{}'.format(i+1)), encoding='UTF-8') as f:
            for line in f.readlines():
                ids, rel = line.strip().split('\t')
                id2rel[int(ids)] = rel

        with open(os.path.join(loc, 'ent_ids_{}'.format(i+1)), encoding='UTF-8') as f:
            if i==0: # kb1
                if trans == True:
                    ent_name_trans = {}
                    with open(os.path.join(loc, 'translated_google.txt'), encoding='UTF-8') as f2:
                        for line1, line2 in zip(f.readlines(), f2.readlines()):
                            ids, ent = line1.strip().split('\t')
                            ent_trans = line2.strip()
                            id2ent[int(ids)] = ent
                            ent_trans = ent_trans.replace('\n','\\n').replace('\t','\\t').replace('\r','').replace('\\"',"'").replace("\\u0022","'")
                            ent_name_trans[ent] = ent_trans
                            if name: # add entity name as an attribute triple
                                kg1.add((ent, 'EA:label', '"'+ent_trans+'"'))
                else: # no translation
                    for line in f.readlines():
                        ids, ent = line.strip().split('\t')
                        ent_name = ent.split('/')[-1].replace('_', ' ')
                        ent_name = ent_name.replace('\n','\\n').replace('\t','\\t').replace('\r','').replace('\\"',"'").replace("\\u0022","'")
                        id2ent[int(ids)] = ent
                        if name: # add entity name as an attribute triple
                            kg1.add((ent, 'EA:label', '"'+ent_name+'"'))
            else: # kb2
                for line in f.readlines():
                    ids, ent = line.strip().split('\t')
                    ent_name = ent.split('/')[-1].replace('_', ' ')
                    ent_name = ent_name.replace('\n','\\n').replace('\t','\\t').replace('\r','').replace('\\"',"'").replace("\\u0022","'")
                    id2ent[int(ids)] = ent
                    if name: # add entity name as an attribute triple
                        kg2.add((ent, 'EA:label', '"'+ent_name+'"'))

        with open(os.path.join(loc, 'triples_{}'.format(i+1)), encoding='UTF-8') as f:
            for line in f.readlines():
                head, rel, tail = line.strip().split('\t')
                head, rel, tail = int(head), int(rel), int(tail)
                # will add bi-directional edges automatically
                if i==0:
                    kg1.add((id2ent[head], id2rel[rel], id2ent[tail]))
                else:
                    kg2.add((id2ent[head], id2rel[rel], id2ent[tail]))

        # load attributes
        if attr:
            for triple in triplesFromTurtleFile(os.path.join(loc, 'att_triples_{}'.format(i+1))):
                head, attribute, literal = triple
                literal = literal.replace('\n','\\n').replace('\t','\\t').replace('\r','').replace('\\"',"'").replace("\\u0022","'")
                # Preprocess literals
                if literal.startswith('"'):
                    # Datatypes
                    if '^^' in literal:
                        str_value, datatype = literal.split('^^')
                        prefixed_datatype = re.sub(r'<http://www\.w3\.org/2001/XMLSchema#([a-zA-Z0-9_-]+)>', r'xsd:\1', datatype)
                        literal = '"'+str_value[1:-1]+'"^^'+prefixed_datatype
                        if i == 0:
                            kg1.add((head[1:-1], attribute[1:-1], literal))
                        else:
                            kg2.add((head[1:-1], attribute[1:-1], literal))
                        continue
                    # other situation (not '^^')
                    if i == 0:
                        kg1.add((head[1:-1], attribute[1:-1], literal))
                    else:
                        kg2.add((head[1:-1], attribute[1:-1], literal))
                else: # literal values lack ""
                    if i == 0:
                        kg1.add((head[1:-1], attribute[1:-1], '"'+literal+'"'))
                    else:
                        kg2.add((head[1:-1], attribute[1:-1], '"'+literal+'"'))
    return kg1, kg2


def load_oaei(loc, format='ttl'):
    """ Loads OAEI datasets """
    if format not in ['ttl', 'xml']:
        raise ValueError("Unsupported format. Please use 'ttl' or 'xml'.")
    if format != 'ttl':
        # For original XML files, we need to convert them to Turtle first
        from rdflib import Graph as RDFGraph
        g = RDFGraph()
        g.parse(os.path.join(loc, 'source.xml'), format='xml')
        g.serialize(os.path.join(loc, 'source.ttl'), format='turtle', encoding='utf-8')
        g = RDFGraph()
        g.parse(os.path.join(loc, 'target.xml'), format='xml')
        g.serialize(os.path.join(loc, 'target.ttl'), format='turtle', encoding='utf-8')
    kg1 = graphFromTurtleFile(os.path.join(loc, 'source.ttl'), message="\nLoading OAEI KB1")
    kg2 = graphFromTurtleFile(os.path.join(loc, 'target.ttl'), message="Loading OAEI KB2")
    return kg1, kg2