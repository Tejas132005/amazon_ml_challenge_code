"""Normalisation behaviour on noise patterns observed in the training data."""
import pyarrow as pa
import pytest

from src.preprocessing import (canon_number, fix_leet, normalize_addresses, normalize_names, skeleton,
                               transliterate_indic)


def names(values):
    out = normalize_names(pa.array(values))
    return {k: v.to_pylist() for k, v in out.items()}


def addrs(values, countries):
    out = normalize_addresses(pa.array(values), pa.array(countries))
    return {k: v.to_pylist() for k, v in out.items()}


@pytest.mark.parametrize("a,b", [
    ("Shaan Project LLP", "5HAAN PROJECT LLP"),
    ("Regional Association", "Regi0nal Ássociation Ássociation"),
    ("Dynamic Impex Private Limited", "Dynamic Impex Private (Limited)"),
    ("Mumbai Financial Pvt Ltd", "*** Mumbai  Financial Pvt Limited"),
    ("Safe Supply International", "Safe Súpply Ínternational"),
    ("Sai Tech Private Limited", "Sai Limited Private Tech"),
    ("Moncada Learning Center LLC", "LLC Moncada Léarning Center"),
])
def test_core_equal_after_normalisation(a, b):
    r = names([a, b])
    assert set(r["name_core"][0].split()) == set(r["name_core"][1].split())


@pytest.mark.parametrize("a,b", [
    ("Style Multi Pvt. Ltd.", "stylemulti.com"),
    ("Gulf Highland Bold LLC", "gulfhighlandbold.com"),
])
def test_domain_names_match_compact(a, b):
    r = names([a, b])
    assert r["name_compact"][0] == r["name_compact"][1]


@pytest.mark.parametrize("latin,native", [
    ("Global Consultants Private Limited", "ग्लोबल कंसल्टेंट्स प्राइवेट लिमिटेड"),
    ("White Products Private Limited", "व्हाइट प्रोडक्ट्स प्राइवेट लिमिटेड"),
    ("Red Agro Private Limited", "রেড অ্যাগ্রো প্রাইভেট লিমিটেড"),
    ("Shree Estate", "శ్రీ ఎస్టేట్"),
    ("Sai Tech Private Limited", "साईं टेक प्राइवेट लिमिटेड"),
])
def test_cross_script_skeleton_and_legal(latin, native):
    r = names([latin, native])
    assert r["name_skel"][0] == r["name_skel"][1]
    assert set(r["name_legal"][0].split()) == set(r["name_legal"][1].split())
    assert r["name_nonlatin"] == [False, True]


def test_distinct_names_stay_distinct():
    r = names(["Sai Tech Private Limited", "Sai It Private Limited", "Blue Ministries", "Blue Minerals"])
    assert len(set(r["name_core"])) == 4


def test_dba_and_noise():
    r = names(["Ectolumdrex dba X+ Madison Inc", "Cardiology Heartland Care Associates #98825", "NA", ""])
    assert r["name_alt_core"][0] == "x madison"
    assert "98825" not in r["name_core"][1] and "98825" in r["name_norm"][1]
    assert r["name_core"][2] == "na" and r["name_core"][3] == ""


def test_leet_rules():
    assert fix_leet("8lue") == "blue" and fix_leet("ujjawa1") == "ujjawal"
    assert fix_leet("2nd") == "2nd" and fix_leet("3m") == "3m" and fix_leet("105") == "105"


def test_transliteration_and_skeleton():
    assert transliterate_indic("ग्लोबल") == "global"
    assert skeleton("private") == skeleton(transliterate_indic("प्राइवेट")) == "prpt"
    assert skeleton("12345") == "12345"


def test_numbers():
    assert canon_number("00115") == "115" and canon_number("no.41") == "41" and canon_number("#306") == "306"
    assert canon_number("3-110/6/g-3") == "3-110/6/g-3" and canon_number("2nd") == "" and canon_number("road") == ""


def test_addresses():
    r = addrs(["115 Willow Point Drive, Little Rock, AR", "00115 Willow Point Dr, PO Box 6009, Little Rock, Arkansas",
               "2/1A, CALCUTTA, HOWRAH, পশ্চিমবঙ্গ", "41, CHENNAI CITY REGION, தமிழ்நாடு",
               "FLAT NO. 4, KOLHAPUR, महाराष्ट्र", "12 Rue de la Paix, 75002 Paris", ""],
              ["us", "us", "india", "india", "india", "france", "us"])
    assert r["addr_core"][0] == r["addr_core"][1] == "115 willow point dr little rock ar" or \
        set(r["addr_core"][0].split()) <= set(r["addr_core"][1].split()) | {"ar", "arkansas"}
    assert r["addr_numbers"][:2] == ["115", "115"]            # PO box number removed
    assert r["addr_region"] == ["ar", "ar", "wb", "tn", "mh", "", ""]
    assert "kolkata" in r["addr_core"][2].split()
    assert r["addr_numbers"][2] == "2/1a" and r["addr_numbers"][5] == "12 75002"
    assert r["addr_empty"] == [False] * 6 + [True]


def test_translit_llp_and_null_token():
    r = names(["Classic Projects LLP", "क्लासिक प्रोजेक्ट्स एलएलपी"])
    assert r["name_legal"] == ["llp", "llp"]
    a = addrs(["UP, NULL, Ghaziabad, 983"], ["india"])
    assert "null" not in a["addr_core"][0].split()
