"""
This file is part of FLORA, an unsupervised system for automatic knowledge graph (KG) alignment. 
The file is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0) by Yiwen Peng, Thomas Bonald, Fabian Suchanek and Lingyun Huang.

Description: Shared literal parsing, normalization, filtering, and streaming helpers for literal processing.
"""

import re
import unicodedata
from urllib.parse import unquote


# Regex for literals
literalRegex = re.compile(r'"((?:\\.|[^"\\])*)"(@([a-z-]+))?(\^\^(.*))?')

# Regex for int values
intRegex = re.compile('^"?[+-]?[0-9]+"?$')

# Regex for float values
floatRegex = re.compile('^"?([+-])?([0-9.]+)"?$')
sciFloatRegex = re.compile('^"?([+-])?([0-9.]+[Ee][+-]?[0-9]+)"?$')

# Regex for numbers: post code, phone number, etc.
# A normalized-number candidate must contain at least one digit.
numberRegex = re.compile(r'(?=.*\d)[\d\W]+')
identifierRegex = re.compile(r'([a-zA-Z]+)(\d+)') # TBD if needed

DATE_DATATYPES = {'xsd:date', 'xsd:gYear', 'xsd:gYearMonth', 'xsd:dateTime', 'xsd:gMonthDay'}

# XSD numeric datatypes describe representation/range, rather than a physical
# unit. Values using any of these datatypes are viewed as dimensionless quantities.
XSD_NUMERIC_DATATYPES = {
    'xsd:decimal', 'xsd:double', 'xsd:float',
    'xsd:integer', 'xsd:long', 'xsd:int', 'xsd:short', 'xsd:byte',
    'xsd:nonNegativeInteger', 'xsd:positiveInteger',
    'xsd:nonPositiveInteger', 'xsd:negativeInteger',
    'xsd:unsignedLong', 'xsd:unsignedInt', 'xsd:unsignedShort',
    'xsd:unsignedByte',
}

# Unit local name -> (physical dimension, multiplier, offset).
# Canonical value = value * multiplier + offset. Local-name lookup supports
# equivalent vocabularies without tying this logic to a dataset or namespace.
NUMERIC_UNIT_DEFINITIONS = {
    # length (metre)
    'metre': ('length', 1.0, 0.0), 'meter': ('length', 1.0, 0.0),
    'kilometre': ('length', 1_000.0, 0.0), 'kilometer': ('length', 1_000.0, 0.0),
    'centimetre': ('length', 0.01, 0.0), 'centimeter': ('length', 0.01, 0.0),
    'millimetre': ('length', 0.001, 0.0), 'millimeter': ('length', 0.001, 0.0),
    'micrometre': ('length', 1e-6, 0.0), 'micrometer': ('length', 1e-6, 0.0),
    'nanometre': ('length', 1e-9, 0.0), 'nanometer': ('length', 1e-9, 0.0),
    'inch': ('length', 0.0254, 0.0), 'foot': ('length', 0.3048, 0.0),
    'yard': ('length', 0.9144, 0.0), 'mile': ('length', 1609.344, 0.0),
    'nauticalmile': ('length', 1852.0, 0.0),
    'nautialmile': ('length', 1852.0, 0.0),  # DBpedia legacy spelling
    'rod': ('length', 5.0292, 0.0),
    'astronomicalunit': ('length', 149_597_870_700.0, 0.0),

    # area (square metre)
    'squaremetre': ('area', 1.0, 0.0), 'squaremeter': ('area', 1.0, 0.0),
    'squarekilometre': ('area', 1e6, 0.0), 'squarekilometer': ('area', 1e6, 0.0),
    'squarefoot': ('area', 0.09290304, 0.0),
    'squaremile': ('area', 2_589_988.110336, 0.0),
    'hectare': ('area', 10_000.0, 0.0), 'acre': ('area', 4046.8564224, 0.0),

    # volume (cubic metre)
    'cubicmetre': ('volume', 1.0, 0.0), 'cubicmeter': ('volume', 1.0, 0.0),
    'cubickilometre': ('volume', 1e9, 0.0), 'cubickilometer': ('volume', 1e9, 0.0),
    'cubiccentimetre': ('volume', 1e-6, 0.0), 'cubiccentimeter': ('volume', 1e-6, 0.0),
    'litre': ('volume', 0.001, 0.0), 'liter': ('volume', 0.001, 0.0),
    'megalitre': ('volume', 1_000.0, 0.0), 'megaliter': ('volume', 1_000.0, 0.0),
    'gigalitre': ('volume', 1_000_000.0, 0.0), 'gigaliter': ('volume', 1_000_000.0, 0.0),

    # time (second) and mass (kilogram)
    'second': ('time', 1.0, 0.0), 'minute': ('time', 60.0, 0.0),
    'hour': ('time', 3600.0, 0.0), 'day': ('time', 86400.0, 0.0),
    'kilogram': ('mass', 1.0, 0.0), 'gram': ('mass', 0.001, 0.0),
    'milligram': ('mass', 1e-6, 0.0), 'tonne': ('mass', 1_000.0, 0.0),
    'pound': ('mass', 0.45359237, 0.0), 'stone': ('mass', 6.35029318, 0.0),

    # speed (metre per second)
    'metrepersecond': ('speed', 1.0, 0.0), 'meterpersecond': ('speed', 1.0, 0.0),
    'kilometreperhour': ('speed', 1.0 / 3.6, 0.0),
    'kilometerperhour': ('speed', 1.0 / 3.6, 0.0),
    'kilometrepersecond': ('speed', 1_000.0, 0.0),
    'kilometerpersecond': ('speed', 1_000.0, 0.0),
    'mileperhour': ('speed', 0.44704, 0.0),
    'footperminute': ('speed', 0.00508, 0.0),
    'knot': ('speed', 1852.0 / 3600.0, 0.0),

    # frequency, pressure, power, energy, force and electrical quantities
    'hertz': ('frequency', 1.0, 0.0), 'kilohertz': ('frequency', 1e3, 0.0),
    'megahertz': ('frequency', 1e6, 0.0), 'gigahertz': ('frequency', 1e9, 0.0),
    'bar': ('pressure', 1e5, 0.0), 'millibar': ('pressure', 100.0, 0.0),
    'watt': ('power', 1.0, 0.0), 'kilowatt': ('power', 1e3, 0.0),
    'megawatt': ('power', 1e6, 0.0), 'horsepower': ('power', 745.6998715822702, 0.0),
    'joule': ('energy', 1.0, 0.0), 'kilojoule': ('energy', 1e3, 0.0),
    'newton': ('force', 1.0, 0.0), 'meganewton': ('force', 1e6, 0.0),
    'kilogramforce': ('force', 9.80665, 0.0), 'pond': ('force', 0.00980665, 0.0),
    'millipond': ('force', 9.80665e-6, 0.0),
    'ampere': ('electric_current', 1.0, 0.0), 'volt': ('voltage', 1.0, 0.0),

    # temperature (kelvin); these require affine conversion.
    'kelvin': ('temperature', 1.0, 0.0),
    'degreecelsius': ('temperature', 1.0, 273.15),
    'degreefahrenheit': ('temperature', 5.0 / 9.0, 255.3722222222222),
    'degreerankine': ('temperature', 5.0 / 9.0, 0.0),

    # information size (bit)
    'bit': ('information', 1.0, 0.0), 'byte': ('information', 8.0, 0.0),
    'kilobyte': ('information', 8_000.0, 0.0),
    'megabyte': ('information', 8_000_000.0, 0.0),
    'gigabyte': ('information', 8_000_000_000.0, 0.0),

    # Ratios are separate from bare numbers: 1% must not equal numeric 1.
    'percent': ('ratio', 0.01, 0.0), 'permil': ('ratio', 0.001, 0.0),

    # Compound quantities found in public KGs.
    'cubicmetrepersecond': ('volume_flow_rate', 1.0, 0.0),
    'grampercubiccentimetre': ('density', 1_000.0, 0.0),
    'inhabitantspersquaremile': ('population_density', 1.0 / 2.589988110336, 0.0),
}


def isLiteral(term):
    return re.match(literalRegex, term) or re.match(floatRegex, term)


def normalize_datatype(datatype):
    """Normalize the datatype to a standard format. e.g., "http://www.w3.org/2001/XMLSchema#string" -> "xsd:string"."""
    if datatype is None:
        return None
    datatype = datatype.strip()
    if datatype.startswith('<') and datatype.endswith('>'):
        datatype = datatype[1:-1]
    xmlschema_prefix = 'http://www.w3.org/2001/XMLSchema#'
    if datatype.startswith(xmlschema_prefix):
        datatype = 'xsd:' + datatype[len(xmlschema_prefix):]
    return datatype


def datatype_local_name(datatype):
    """Return a case-insensitive local name for a datatype URI or QName."""
    datatype = normalize_datatype(datatype)
    if datatype is None:
        return None
    local_name = datatype.rsplit('#', 1)[-1].rsplit('/', 1)[-1].rsplit(':', 1)[-1]
    return re.sub(r'[^a-z0-9]', '', local_name.lower())


def numeric_quantity_key(value, datatype):
    """Return ``(dimension, canonical_value)`` for a parsed numeric literal.

    Unknown custom datatypes use their normalized datatype as the dimension
    signature. This permits same-datatype comparisons but prevents unsafe
    cross-type matches. Untyped numbers return ``None`` because FLORA handles
    them as digit-like strings rather than quantities.
    """
    if not isinstance(value, (int, float)) or datatype is None:
        return None

    datatype = normalize_datatype(datatype)
    if datatype in XSD_NUMERIC_DATATYPES:
        return ('dimensionless', value)

    definition = NUMERIC_UNIT_DEFINITIONS.get(datatype_local_name(datatype))
    if definition is not None:
        dimension, multiplier, offset = definition
        return (dimension, value * multiplier + offset)

    return ('datatype:' + datatype.lower(), value)


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


def numeric_normalization(term):
    term = term.strip('"')
    # Normalize the input string by removing all non-digit characters, except the sign
    # sign = term[0] if term.startswith('-') or term.startswith('+') else None
    normalized = re.sub(r'[^0-9]', '', term)
    return normalized


def splitLiteral(term):
    """Return text, numeric value, language, and datatype, preserving text escapes."""
    literal, _, lang, _, datatype = re.match(literalRegex, term).groups()
    datatype = normalize_datatype(datatype)
    # Dates
    if datatype in DATE_DATATYPES:
        return (literal, None, lang, datatype)

    # Numbers
    floatmatch = re.match(floatRegex, literal)
    scifloatmatch = re.match(sciFloatRegex, literal)
    if (floatmatch or scifloatmatch) and lang is None:
        try:
            # some identifiers are integers
            Value = int(literal.strip('"'))
            if len(str(Value)) != len(literal.strip('"')):
                # e.g., "06" /= 6, "+3" /= 3
                return (literal, None, lang, datatype) # datatype 'none'
            return (literal, Value, lang, datatype)
        except:
            try:
                # e.g., code version 1.0 /= 1
                Value = float(literal.strip('"'))
                return (literal, Value, lang, datatype)
            except ValueError:
                # e.g. "23.78.9" version
                return (literal, None, lang, datatype)

    matchNumber = re.fullmatch(numberRegex, literal)
    # Check if the string is a number type, e.g., "818/762-1221"
    if matchNumber:
        Value = numeric_normalization(literal)
        if len(Value) > 0:
            return (literal, 'normalized_' + Value, lang, datatype)
    # Strings: lowecasing, order-agnostic, decode unicode
    # Pre-processing for string literals
    de_literal = decode_unicode(literal)
    return (de_literal, None, lang, 'xsd:string')


def is_punctuation_only_literal(txt):
    """Check if a string literal consists only of punctuation characters."""
    txt = decode_unicode(txt).strip()
    return bool(txt) and not any(char.isalnum() for char in txt)


def is_human_readable(txt):
    if not txt or len(txt) == 0:
        return False
    # not a web link
    if txt.startswith('http'):
        return False
    non_alpha_ratio = sum(not c.isalpha() for c in txt) / len(txt)
    return non_alpha_ratio < 0.5


def is_faiss_literal_candidate(txt):
    """Filter string literals before FAISS search."""
    if not txt:
        return False
    txt = decode_unicode(txt).strip()
    if not txt:
        return False
    return is_human_readable(txt)


def normalize_string_literal(value):
    value = decode_unicode(value) # decode unicode characters
    value = unquote(value) # decode URL-encoded characters
    value = value.strip() # remove spaces
    value = value.lower() # convert to lowercase

    # Normalize common separators.
    value = value.replace('_', ' ') # replace underscores with spaces
    value = re.sub(r'\s+', ' ', value) # replace multiple spaces with a single space
    return value.strip()


def is_latin_alpha(char):
    if not char.isalpha():
        return False
    try:
        return 'LATIN' in unicodedata.name(char)
    except ValueError:
        return False


def is_english_string_literal(literal, lang=None, min_latin_ratio=0.8):
    """Check if a string literal is likely to be English based on its language tag and the ratio of Latin letters."""
    if lang is not None:
        lang = lang.lower()
        if lang != 'en' and not lang.startswith('en-'):
            return False

    letters = [char for char in decode_unicode(literal) if char.isalpha()]
    if not letters:
        return True

    latin_letters = sum(1 for char in letters if is_latin_alpha(char))
    return latin_letters / len(letters) >= min_latin_ratio


def iter_literal_objects(kb):
    if hasattr(kb, 'iter_literal_object_ids') and hasattr(kb, 'entity_for_id'): # for CompactGraph
        objects = (kb.entity_for_id(object_id) for object_id in kb.iter_literal_object_ids())
    else:
        objects = kb.objects()
    yield from (object for object in objects if isLiteral(object))


def iter_ttl_literal_triples(path, fast_line_parser=False):
    """Yield (subject, predicate, literal_object) from TTL files.
    Uses FLORA's Turtle parser by default.
    Set fast_line_parser=True for a faster scanner for one-triple-per-line files.
    """
    if not fast_line_parser:
        import utils

        for subject, predicate, obj in utils.triplesFromTurtleFile(path):
            if isLiteral(obj):
                yield subject, predicate, obj
        return

    with open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("@prefix") or line.startswith("#"):
                continue
            parts = line.split(None, 2)
            if len(parts) != 3 or not parts[2].endswith("."):
                continue
            obj = parts[2][:-1].strip()
            match = literalRegex.match(obj)
            if match is not None:
                # Use the same literal representation as utils.termsAndSeparators.
                literal = match.group(1)
                literal = literal.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '')
                literal = literal.replace('\\"', "'").replace('\\u0022', "'")
                obj = '"' + literal + obj[match.end(1):]
                yield parts[0], parts[1], obj


def iter_ttl_literal_facts(path, fast_line_parser=False):
    """Yield (subject, literal_object) from TTL files."""
    for subject, _, obj in iter_ttl_literal_triples(path, fast_line_parser=fast_line_parser):
        yield subject, obj


def iter_ttl_object_literals(path, fast_line_parser=False):
    """Yield literal objects from TTL files using the selected parser."""
    for _, obj in iter_ttl_literal_facts(path, fast_line_parser=fast_line_parser):
        yield obj