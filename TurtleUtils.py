import gzip
import os
import codecs
import re
import sys
from io import StringIO
# import Prefixes
from collections import defaultdict
from functools import reduce

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
        self.index={}
        self.relindex={} # {predicate:{object:subject}}
        return
    def add(self, triple):
        (subject, predicate, obj) = triple
        if subject not in self.index:
            self.index[subject]={}
        m=self.index[subject]
        if predicate not in m:
            m[predicate]=set()
        m[predicate].add(obj)

        # relindex
        if predicate not in self.relindex:
            self.relindex[predicate]={}
        m=self.relindex[predicate]
        if subject not in m:
            m[subject]=set()
        m[subject].add(obj)
        
        if not isInverse(predicate):
            self.add((obj,invert(predicate),subject))
        self.pred2num=None
    def remove(self, triple):
        (subject, predicate, obj) = triple
        if subject not in self.index:
            return
        m=self.index[subject]
        if predicate not in m:
            return
        m[predicate].discard(obj)
        if len(m[predicate])==0:
            self.index[subject].pop(predicate)
            if len(self.index[subject])==0:
                self.index.pop(subject)
        if not isInverse(predicate):
            self.remove((obj,invert(predicate),subject))
        self.pred2num=None
    def __contains__(self, triple):
        (subject, predicate, obj) = triple
        if subject not in self.index:
            return False
        m=self.index[subject]
        if predicate not in m:
            return False
        return obj in m[predicate]
    def __iter__(self):
        for s in self.index:
            for p in self.index[s]:
                for o in self.index[s][p]:
                    yield (s,p,o)
    def loadTurtleFile(self, file, message=None):
        for triple in triplesFromTurtleFile(file, message):
            self.add(triple)
    def getList(self, listStart):
        """ Returns the elements of an RDF list"""
        result=[]
        while listStart and listStart!='rdf:nil':
            result.extend(self.index[listStart].get('rdf:first',[]))
            if 'rdf:rest' not in self.index[listStart]:
                break
            listStart=list(self.index[listStart]['rdf:rest'])[0]            
        return result
    def predicates(self):
        if not self.pred2num:
            self.numFactsWithPredicate("blah")
        return self.pred2num
    def attributes(self):
        """ Returns all the attributes of the graph """
        result=set()
        for predicate in self.relindex:
            if self.isAttribute(predicate):
                result.add(predicate)
                result.add(invert(predicate)) # add inverse
        return result
    def numFactsWithPredicate(self, predicate):
        if self.pred2num:
            return self.pred2num[predicate] if predicate in self.pred2num else 0
        self.pred2num={}
        for subject in self.index:
            for pred in self.index[subject]:
                if pred not in self.pred2num:
                    self.pred2num[pred]=0
                self.pred2num[pred]+=len(self.index[subject][pred])
        return self.numFactsWithPredicate(predicate)
    # def localFunctionality(self, subject, pred):
    #     return 1.0/len(self.index[subject][pred])
    def isAttribute(self, pred):
        """ Returns TRUE if the predicate is an attribute"""
        # check if there are literals in the object
        if pred in self.relindex:
            for literal in self.relindex[pred]:
                if isLiteral(literal):
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
            if commonObjs is None:
                commonObjs = self.index[subjects[i]][preds[i]]
            else:
                commonObjs = commonObjs & self.index[subjects[i]][preds[i]]
        try:
            value = 1.0/len(commonObjs)
        except ZeroDivisionError:
            value = 0
        return value
    def objects(self, subject=None, predicate=None):
        # We create a copy here instead of using a generator
        # because the user loop may want to change the graph
        result=[]
        if subject and subject not in self.index:
            return result
        for s in ([subject] if subject else self.index):
            for p in ([predicate] if predicate else self.index[s]):
                if p in self.index[s]:
                    result.extend(self.index[s][p])
        return set(result)
    def subjects(self, predicate=None, object=None):        
        pred = invert(predicate) if predicate else None 
        return self.objects(subject=object, predicate=pred)
        # return self.objects(subject=object, predicate=invert(predicate))    
    def triplesWithObject(self, obj, predicates=[]):
        return self.triplesWithSubject(obj, [invert(p) for p in predicates])
    def triplesWithSubject(self, subject, predicates=[]):
        for predicate in predicates if len(predicates) else self.index[subject].keys():
            if predicate in self.index[subject]:
                for obj in self.index[subject][predicate]:
                    yield (subject, predicate, obj)
    def triplesWithPredicate(self, *predicates):
        for subject in self.index:
            for predicate in predicates:
                if predicate in self.index[subject]:
                    for object in self.index[subject][predicate]:
                        yield (subject, predicate, object)
    def headTriplesWithPredicateList(self, predicatesWithCount):
        """ Returns the triples dictionary where the head entity 
            has all predicates in the given predicates """
        result = {} # {head: pred: tail}
        heads = [set(self.relindex[pred].keys()) for pred in predicatesWithCount]
        sharedHeads = reduce(lambda x, y: x & y, heads)
         
        for predicate in set(predicatesWithCount):
            for subject in sharedHeads:
                if len(self.relindex[predicate][subject]) < predicatesWithCount[predicate]:
                    continue
                if subject not in result:
                    result[subject] = set()
                # To avoid high complexity for type, genre relations, 
                # which have a super hugh number of subjects
                pred2count_ = defaultdict(int)
                for obj in self.relindex[predicate][subject]:
                    result[subject].add((subject, predicate, obj))
                    pred2count_[predicate] += 1
                    if pred2count_[predicate] > 10:
                        break
                # result[subject][predicate].update(self.relindex[predicate][subject])
        return result
    def printToWriter(self, result):        
        for subject in self.index:
            if subject.startswith("_:list_"):
                continue
            result.write("\n")
            result.write(subject)
            result.write(' ')
            hasPreviousPred=False
            for predicate in self.index[subject]:
                if isInverse(predicate):
                    continue
                if hasPreviousPred:
                    result.write(' ;\n\t')
                hasPreviousPred=True            
                result.write(predicate)
                result.write(' ')
                hasPrevious=False
                for obj in self.index[subject][predicate]:                    
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
    # def printToFile(self, file):
    #     with open(file, "wt", encoding="utf-8") as out:
    #         for p in Prefixes.prefixes:
    #             out.write("@prefix "+p+": <"+Prefixes.prefixes[p]+"> .\n")
    #         self.printToWriter(out)
    def __str__(self):
        buffer=StringIO()
        buffer.write("# RDF Graph\n")
        self.printToWriter(buffer)
        return buffer.getvalue()
    def someSubject(self):
        for key in self.index:
            return key
        return None
    def __len__(self):
        # Total number of facts
        count=0
        for p in self.predicates():
            count+=self.numFactsWithPredicate(p)
        return count
        

# Regex for literals
literalRegex=re.compile('"([^"]*)"(@([a-z-]+))?(\\^\\^(.*))?')

# Regex for int values
intRegex=re.compile('^"?[+-]?[0-9.]+"?$')

def isLiteral(term):
    return re.match(literalRegex,term) or re.match(intRegex,term)

