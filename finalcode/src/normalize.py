"""Record normalisation: names and addresses -> canonical, comparable strings.

All rules are generic, language-agnostic string rules (abbreviation tables, legal
forms, honorifics) plus the transliteration dictionary learned from training data.
"""
import re

import numpy as np
import pandas as pd
from unidecode import unidecode

from translit import Transliterator

# ----------------------------------------------------------------------------- names
ALIAS_RE = re.compile(
    r"^(.*?)\s+(?:d/?b/?a:?|doing business as|formerly(?: known as)?|f/?k/?a|a/?k/?a|aka|"
    r"trading as|t/a)\s*:?\s+(.+)$", re.I)
PIPE_RE = re.compile(r"\s*\|\s*")
URL_RE = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|in|co\.in|fr|biz|info|co)\b")
ORDINAL_RE = re.compile(r"^\d+(?:st|nd|rd|th|er|e|eme)$")
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b",
                      "@": "a", "$": "s"})
NONWORD_RE = re.compile(r"[^a-z0-9& ]+")
SPACE_RE = re.compile(r"\s+")

# legal / entity-form tokens (all countries).  Removed from the "core" name.
LEGAL = {
    "private", "pvt", "pvt.", "limited", "ltd", "llc", "l", "c", "inc", "incorporated", "corp",
    "corporation", "co", "company", "llp", "lp", "pc", "pllc", "plc", "pa", "p", "ltda",
    "sarl", "sas", "sasu", "sa", "sci", "eurl", "snc", "scop", "scp", "selarl", "gie",
    "gmbh", "opc", "pte", "trust", "cie", "et",
}
# honorifics / junk words that are prepended as noise
PREFIX = {"the", "smt", "ms", "m/s", "sri", "shri", "shree", "mr", "mrs", "dr", "a", "le", "la", "les"}
AMP = {"&": "and", "et": "and"}


def _fix_token(t):
    """Undo leetspeak (e.g. MUM8AI -> mumbai) in tokens that mix letters and digits; plain numbers and ordinals are kept."""
    if t.isdigit() or ORDINAL_RE.match(t):
        return t
    if any(ch.isdigit() for ch in t) and any(ch.isalpha() for ch in t):
        return t.translate(LEET)
    return t


def _basic(s):
    """ASCII-fold and lowercase a string and collapse dotted legal forms (l.l.c. -> llc, s.a.r.l -> sarl, ...)."""
    s = unidecode(s).lower()
    s = s.replace("l.l.c.", "llc").replace("l.l.p.", "llp").replace("p.c.", "pc").replace(
        "s.a.s", "sas").replace("s.a.r.l", "sarl").replace("pvt.", "pvt ").replace("ltd.", "ltd ")
    return s


def normalize_name(raw, tr=None):
    """Return (full, core, alias_core) for a business name."""
    s = raw or ""
    if tr is not None:
        s = tr.name(s)
    s = _basic(s)
    s = PIPE_RE.split(s)[0] if "|" in s and not s.startswith("|") else s
    alias = ""
    m = ALIAS_RE.match(s)
    if m:
        alias, s = m.group(1), m.group(2)  # the real (older/legal) name follows the keyword
    url = URL_RE.search(s)
    if url and len(s.split()) == 1:
        s = url.group(1).replace("-", " ")
    s = s.replace("#", " ").replace("&", " & ")
    s = NONWORD_RE.sub(" ", s)
    toks = [_fix_token(t) for t in s.split()]
    toks = ["and" if t in AMP else t for t in toks]
    full = " ".join(toks)
    core = [t for t in toks if t not in LEGAL and t != "and"]
    while len(core) > 1 and core[0] in PREFIX:
        core = core[1:]
    return full, " ".join(core), _alias_core(alias)


def _alias_core(a):
    """Core form of an alias name: punctuation removed and legal-form tokens dropped."""
    if not a:
        return ""
    a = NONWORD_RE.sub(" ", a.replace("&", " "))
    return " ".join(t for t in a.split() if t not in LEGAL)


# -------------------------------------------------------------------------- addresses
ABBR = {
    # street types (US / India / France) -> canonical short form
    "street": "st", "str": "st", "saint": "st", "st": "st", "ste": "ste", "sainte": "ste",
    "avenue": "ave", "av": "ave", "ave": "ave", "avn": "ave",
    "road": "rd", "rd": "rd", "rte": "rte", "route": "rte",
    "drive": "dr", "dr": "dr", "drv": "dr",
    "lane": "ln", "ln": "ln", "court": "ct", "ct": "ct", "circle": "cir", "cir": "cir",
    "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "bld": "blvd", "boul": "blvd",
    "place": "pl", "pl": "pl", "parkway": "pkwy", "pkwy": "pkwy", "highway": "hwy", "hwy": "hwy",
    "terrace": "ter", "ter": "ter", "trail": "trl", "trl": "trl", "square": "sq", "sq": "sq",
    "mount": "mt", "mt": "mt", "fort": "ft", "ft": "ft", "point": "pt", "pt": "pt",
    "suite": "unit", "apartment": "unit", "apt": "unit", "unit": "unit", "room": "unit",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw",
    "rue": "rue", "r": "rue", "chemin": "che", "che": "che", "ch": "che", "impasse": "imp",
    "imp": "imp", "allee": "all", "all": "all", "faubourg": "fg", "fg": "fg", "quai": "quai",
    "sector": "sec", "sec": "sec", "nagar": "ngr", "ngr": "ngr", "colony": "col", "col": "col",
    "marg": "marg", "cross": "crs", "crs": "crs", "main": "mn", "mn": "mn",
    "floor": "flr", "flr": "flr", "building": "bldg", "bldg": "bldg", "near": "nr", "nr": "nr",
    "opposite": "opp", "opp": "opp", "behind": "bhd",
    "po": "po", "box": "box", "district": "dist", "dist": "dist",
}
# filler tokens carrying no location information
ADDR_STOP = {"no", "number", "h", "hno", "house", "door", "plot", "flat", "shop", "office",
             "block", "of", "the", "de", "du", "des", "la", "le", "les", "d", "l", "and",
             "null", "none", "nan", "na", "city", "town", "village", "and"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc", "puerto rico": "pr",
}
IN_STATES = {
    "maharashtra": "mh", "delhi": "dl", "new delhi": "dl", "uttar pradesh": "up", "karnataka": "ka",
    "tamil nadu": "tn", "andhra pradesh": "ap", "telangana": "ts", "west bengal": "wb",
    "gujarat": "gj", "rajasthan": "rj", "madhya pradesh": "mp", "haryana": "hr", "punjab": "pb",
    "kerala": "kl", "keralam": "kl", "tg": "ts", "orissa": "od", "odisha": "od", "bihar": "br", "jharkhand": "jh",
    "chhattisgarh": "cg", "assam": "as", "goa": "ga", "uttarakhand": "uk", "himachal pradesh": "hp",
    "jammu and kashmir": "jk", "jammu & kashmir": "jk", "chandigarh": "ch", "puducherry": "py",
    "pondicherry": "py", "tripura": "tr", "meghalaya": "ml", "manipur": "mn", "nagaland": "nl",
    "mizoram": "mz", "sikkim": "sk", "arunachal pradesh": "ar",
}
# France: regions, plus the departements seen in the data mapped to their region (Source 1
# writes the region, Sources 2/3 the region or the departement).
FR_REGIONS = {
    "hauts-de-france": "hdf", "nord": "hdf", "pas-de-calais": "hdf", "somme": "hdf", "oise": "hdf",
    "aisne": "hdf",
    "nouvelle-aquitaine": "naq", "gironde": "naq", "landes": "naq", "pyrenees-atlantiques": "naq",
    "lot-et-garonne": "naq", "dordogne": "naq", "charente-maritime": "naq",
    "pays de la loire": "pdl", "loire-atlantique": "pdl", "maine-et-loire": "pdl", "vendee": "pdl",
    "sarthe": "pdl", "mayenne": "pdl",
    "ile-de-france": "idf", "grand est": "ges", "normandie": "nor", "bretagne": "bre",
    "centre-val de loire": "cvl", "bourgogne-franche-comte": "bfc", "auvergne-rhone-alpes": "ara",
    "occitanie": "occ", "provence-alpes-cote d'azur": "pac", "provence-alpes-cote dazur": "pac", "corse": "cor",
}
STATE_MAP = {"US": US_STATES, "India": IN_STATES, "France": FR_REGIONS}
STATE_CODES = {c: set(m.values()) for c, m in STATE_MAP.items()}
# Hyderabad records carry both the pre-2014 (Andhra Pradesh) and current (Telangana)
# state in different sources, so the two are treated as one blocking region.
STATE_MERGE = {"ap": "ts"}
# State names that are also city names ("Washington, DC", "Delaware, OH", "Delhi, HR"): when the
# address has another, different state-like part, that other part is the state, in any order.
CITY_LIKE_STATES = {"washington", "delaware", "oregon", "indiana", "nevada", "kansas", "delhi", "new delhi"}
ADDR_NONWORD_RE = re.compile(r"[^a-z0-9/\- ]+")
NUM_RE = re.compile(r"\d+")


def normalize_address(raw, country, tr=None):
    """Return (tokens_string, numbers_string, state) for an address."""
    s = raw or ""
    if tr is not None:
        s = tr.address(s)
    s = unidecode(s).lower()
    comps = [c.strip() for c in s.split(",")]
    smap = STATE_MAP.get(country, {})
    codes = STATE_CODES.get(country, set())
    cleaned = [SPACE_RE.sub(" ", ADDR_NONWORD_RE.sub(" ", c)).strip() for c in comps]
    cleaned = [c for c in cleaned if c]
    # the LAST state-like component is the state ("Washington, DC" -> dc), except that a
    # city-like state name loses against a different state ("DC, Washington" -> dc too);
    # the other state-like components (city names such as Washington / Delhi) stay as tokens
    hits = [(i, STATE_MERGE.get(smap.get(c2, c2), smap.get(c2, c2))) for i, c2 in enumerate(cleaned)
            if c2 in smap or c2 in codes]
    if len({h[1] for h in hits}) > 1:
        hits = [h for h in hits if cleaned[h[0]] not in CITY_LIKE_STATES] or hits
    pos, state = hits[-1] if hits else (-1, "")
    kept = [c for i, c in enumerate(cleaned) if i != pos]
    s = " ".join(kept)
    nums = NUM_RE.findall(s)
    s = s.replace("/", " ").replace("-", " ")
    toks = []
    for t in s.split():
        if t.isdigit():
            continue
        t = ABBR.get(t, t)
        if t in ADDR_STOP:
            continue
        toks.append(t)
    return " ".join(toks), " ".join(nums), state


# ------------------------------------------------------------------------- phonetic
_VOW = str.maketrans("", "", "aeiouy")
_PH = [("ph", "f"), ("ck", "k"), ("q", "k"), ("z", "s"), ("x", "ks"), ("w", "v"), ("c", "k"),
       ("th", "t"), ("dh", "d"), ("bh", "b"), ("kh", "k"), ("gh", "g"), ("sh", "s"), ("h", "")]
DUP_RE = re.compile(r"(.)\1+")


def phonetic(core):
    """Per token: sound substitutions on the whole token, collapse repeated letters, then keep the
    (substituted) first letter plus the consonants.  phuudd -> fd, philip / filip -> flp."""
    out = []
    for t in core.split():
        if t.isdigit():
            out.append(t)
            continue
        r = t
        for a, b in _PH:
            r = r.replace(a, b)
        r = DUP_RE.sub(r"\1", r)
        if r:
            out.append(r[0] + r[1:].translate(_VOW))
    return " ".join(out)


# ---------------------------------------------------------------------------- frame
def normalize_frame(df, tr):
    """Normalise a raw record frame: name (nm, core, alias), address (ad, nums, state), phonetic key and unknown-native-word count."""
    names = [normalize_name(n, tr) for n in df.business_name.values]
    addrs = [normalize_address(a, c, tr) for a, c in zip(df.business_address.values, df.country.values)]
    out = pd.DataFrame({
        "entity_id": df.entity_id.values,
        "country": df.country.values,
        "nm": [x[0] for x in names],
        "core": [x[1] for x in names],
        "alias": [x[2] for x in names],
        "ad": [x[0] for x in addrs],
        "nums": [x[1] for x in addrs],
        "state": [x[2] for x in addrs],
    })
    out["ph"] = [phonetic(c) for c in out.core.values]
    out["nunk"] = np.array([tr.unknown(n) if tr is not None else 0 for n in df.business_name.values], np.int16)
    return out


if __name__ == "__main__":
    import sys, os
    from multiprocessing import Pool
    work = sys.argv[1]
    tr = Transliterator(os.path.join(work, "translit.json"))
    tests = ["Onyxveo formerly Espinoza & Davis Industrials LLC", "Fresh LP Coffee | www.freshlpc.com",
             "MUM8AI LIMTIED PRIVATE LIMITED", "Sri MAGPPIE PARK PVT LTD", "#empiremolecular",
             "लाइफ इन्वेस्टमेंट्स", "Irizeph formerly known as Red Trading Private Limited",
             "The Á Strategic Environmental", "Espinoza & Davis Industrials L.L.C.", "Marina Ecole France Sarl"]
    for t in tests:
        print(t, "->", normalize_name(t, tr))
    for a, c in [("2 Muldowney Circle, Unit Apartment B, Poughkeepsie, NY", "US"),
                 ("MN, RAMSEY, 5425 152ND AVENUE", "US"),
                 ("B-#20, Regency Plaza, Kalyan, Thane, MH", "India"),
                 ("OFFICE NO 212, PLOT NO 20, THE GREAT EASTERNGALLERIA, THANE, महाराष्ट्र", "India"),
                 ("63 R. DE DIEPPE, LILLE, Hauts-de-France", "France")]:
        print(a, "->", normalize_address(a, c, tr))
