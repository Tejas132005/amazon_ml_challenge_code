"""Static, hand-written domain vocabularies used by preprocessing.

These are generic linguistic lists (legal forms, address abbreviations, region
names), not looked-up business data, so they comply with the no-external-data
rule. Keys are lowercase ASCII after accent folding. Region lists are keyed by
lowercased country label; unseen labels (e.g. France) simply get no region
extraction -- nothing else in the pipeline depends on them.
"""
from __future__ import annotations

# ---- Business names ---------------------------------------------------------
# token -> canonical legal-form token (multi-word values allowed)
LEGAL_CANON: dict[str, str] = {
    "private": "pvt", "pvt": "pvt", "pvtltd": "pvt ltd", "prvt": "pvt",
    "limited": "ltd", "ltd": "ltd", "ltda": "ltd",
    "incorporated": "inc", "inc": "inc",
    "corporation": "corp", "corp": "corp", "co": "co", "company": "co", "cos": "co",
    "llc": "llc", "lllc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc",
    "pc": "pc", "pa": "pa", "lc": "lc", "ltdco": "ltd co", "opc": "opc", "pte": "pte", "pty": "pty",
    "cie": "co", "sa": "sa", "sas": "sas", "sasu": "sasu", "sarl": "sarl", "eurl": "eurl", "sci": "sci",
    "snc": "snc", "scop": "scop", "selarl": "selarl", "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv",
}
LEGAL_TOKENS: frozenset[str] = frozenset(t for v in LEGAL_CANON.values() for t in v.split())
# Full words used to build legal-form skeletons for transliterated (non-Latin) tokens.
# Abbreviations are excluded: their skeletons are too short ("pvt" -> "pt" == "white").
LEGAL_WORDS_FOR_SKELETON: tuple[str, ...] = ("private", "limited", "company", "corporation", "incorporated")
# Spelled-out abbreviations after transliteration (एलएलपी -> "elelpi")
TRANSLIT_LEGAL: dict[str, str] = {"elelpi": "llp", "elaelapi": "llp", "elelsi": "llc", "elaelasi": "llc",
                                  "pivitti": "pvt", "limittedd": "ltd"}
NAME_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "of", "dba", "aka", "fka", "ta", "et", "de", "du", "des", "la", "le", "les",
})
# Separators that introduce a trade name: text after them goes to name_alt
DBA_PATTERN = r"\b(?:d\s?/?\s?b\s?/?\s?a|doing business as|a\s?/\s?k\s?/\s?a|aka|t\s?/\s?a|trading as)\b"
TLD_PATTERN = r"\.(?:com|net|org|co\.in|co|in|fr|io|biz|info|us)\b"
# Digit -> letter substitutions for leetspeak typos ("5haan", "8lue", "regi0nal")
LEET: dict[str, str] = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"}

# ---- Addresses --------------------------------------------------------------
ADDR_CANON: dict[str, str] = {
    "street": "st", "str": "st", "saint": "st", "road": "rd", "avenue": "ave", "av": "ave", "avn": "ave",
    "drive": "dr", "drv": "dr", "lane": "ln", "court": "ct", "boulevard": "blvd", "boul": "blvd",
    "bd": "blvd", "circle": "cir", "place": "pl", "highway": "hwy", "parkway": "pkwy", "trail": "trl",
    "terrace": "ter", "square": "sq", "suite": "ste", "apartment": "apt", "apartments": "apt",
    "appartment": "apt", "floor": "fl", "flr": "fl", "building": "bldg", "bldg": "bldg", "number": "no",
    "num": "no", "nr": "near", "opposite": "opp", "mount": "mt", "fort": "ft", "sector": "sec",
    "first": "1st", "ist": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
    "sixth": "6th", "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th",
    "route": "rte", "rue": "r", "appt": "apt", "chemin": "ch", "impasse": "imp", "allee": "all", "sainte": "ste",
    # city name variants (old/new spellings)
    "calcutta": "kolkata", "bombay": "mumbai", "madras": "chennai", "bangalore": "bengaluru",
    "gurgaon": "gurugram", "poona": "pune", "baroda": "vadodara", "trivandrum": "thiruvananthapuram",
    "cochin": "kochi", "mysore": "mysuru", "pondicherry": "puducherry", "orissa": "odisha",
    "calicut": "kozhikode", "vizag": "visakhapatnam", "allahabad": "prayagraj", "ahmadabad": "ahmedabad",
    "banaras": "varanasi", "benaras": "varanasi", "trichy": "tiruchirappalli",
}
# Low-information address tokens: kept in addr_norm, excluded from addr_core
ADDR_STOPWORDS: frozenset[str] = frozenset({
    "no", "h", "hno", "house", "plot", "flat", "shop", "door", "office", "room", "unit", "ste", "apt",
    "fl", "bldg", "near", "opp", "behind", "bh", "the", "of", "and", "ground", "gf", "ugf", "po",
    "dist", "district", "city", "town", "village", "tehsil", "taluk", "cedex", "bis", "ter", "quater", "null", "none", "nil", "de", "la", "du", "des", "le",
})
PO_BOX_PATTERN = r"\b(?:p\s?o\s?box|post office box|pmb|box)\s*#?\s*[0-9]+\b"

US_REGIONS: dict[str, str] = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "district of columbia": "dc", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "puerto rico": "pr", "guam": "gu",
}
INDIA_REGIONS: dict[str, str] = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "chattisgarh": "cg", "ct": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh",
    "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od",
    "or": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn",
    "telangana": "tg", "ts": "tg", "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk",
    "uttaranchal": "uk", "ua": "uk", "west bengal": "wb", "paschim banga": "wb", "paschimbanga": "wb",
    "paschimbang": "wb", "delhi": "dl", "nct of delhi": "dl", "jammu and kashmir": "jk", "jammu kashmir": "jk",
    "ladakh": "la", "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
    "andaman and nicobar islands": "an", "dadra and nagar haveli and daman and diu": "dn",
    "dadra and nagar haveli": "dn", "daman and diu": "dd", "lakshadweep": "ld",
}
REGIONS_BY_COUNTRY: dict[str, dict[str, str]] = {"us": US_REGIONS, "india": INDIA_REGIONS}
