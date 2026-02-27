"""
AI Engine – OpenAI GPT-5.2 powered PII entity detection.

Handles large texts by splitting into chunks that fit within AI token limits
and merging results, ensuring consistent variable assignment across chunks.
"""

import json
import re
from typing import Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
# Processing modes  (used by gui.py and pdf_processor.py too)
# ---------------------------------------------------------------------------

MODE_ANONYMIZE = "anonymize"            # solid black bars, no labels
MODE_PSEUDO_VARS = "pseudo_vars"        # black bars with hex variable labels
MODE_PSEUDO_NATURAL = "pseudo_natural"  # natural-sounding replacement values

# ---------------------------------------------------------------------------
# Intensity  (always maximum – no user choice)
# ---------------------------------------------------------------------------

INTENSITY_HARD = "hard"         # aggressive – in doubt, always redact

# ---------------------------------------------------------------------------
# Scope  (which categories of PII to target)
# ---------------------------------------------------------------------------

SCOPE_NAMES_ONLY = "names_only"  # person-identifying: names, addresses, contact
SCOPE_ALL = "all"                # above + financial numbers, amounts, percentages

# Categories for "Personen-Daten" scope (everything that identifies a person)
_PERSON_CATEGORIES = {
    "VORNAME", "NACHNAME", "STRASSE", "HAUSNUMMER", "STADT", "PLZ", "LAND",
    "EMAIL", "TELEFON", "UNTERNEHMEN", "GEBURTSDATUM", "UNTERSCHRIFT",
    "SOZIALVERSICHERUNG", "AUSWEISNUMMER", "GRUNDSTUECK",
}

# Approximate character limit per chunk.  Most models handle ~120k chars
# comfortably; we stay well below to leave room for the system prompt and
# response.  Overlapping avoids splitting an entity at a boundary.
CHUNK_SIZE = 60_000
CHUNK_OVERLAP = 2_000

# ---------------------------------------------------------------------------
# Prompt that instructs the AI to find all PII entities
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Du bist ein präziser Experte für Datenanonymisierung. Deine Aufgabe ist es, in einem gegebenen Text ALLE personenbezogenen und identifizierenden Daten LÜCKENLOS zu finden.

OBERSTE REGEL: Finde ALLE echten personenbezogenen Daten – aber nur ECHTE PII. Lieber einmal zu viel als zu wenig, ABER: Dokumentstruktur (Nummerierungen, Paragraphen, Gliederungen) darf NIEMALS als PII gemeldet werden. Das Dokument muss nach der Schwärzung noch lesbar und strukturell intakt sein.

Du musst folgende Kategorien erkennen – in ALLEN Sprachen, die im Text vorkommen:

1. VORNAME – Vornamen von Personen. Auch: Spitznamen, Rufnamen, abgekürzte Vornamen (z.B. "Max", "M.", "Hans-Peter", "J.", "Dr. Hans"). JEDER Vorname muss erkannt werden, egal ob er am Satzanfang, in einer Aufzählung, in einer Grußformel, in einer Unterschrift, in einem Briefkopf, in einer E-Mail-Signatur oder irgendwo anders steht. Auch einzelne Buchstaben mit Punkt (z.B. "M."), die als Vornamens-Abkürzung verwendet werden.
2. NACHNAME – Nachnamen von Personen. Auch: Doppelnamen (z.B. "Müller-Schmidt"), Namenszusätze (z.B. "von", "van", "de", "zu" als Teil des Namens), Titel+Name-Kombinationen. JEDER Nachname muss erkannt werden, auch wenn er nur einmal vorkommt. Nachnamen in Firmennamen (z.B. "Müller" in "Kanzlei Müller") EBENFALLS erkennen.
3. STRASSE – Straßennamen (z.B. "Hauptstraße", "Bahnhofstr.", "Am Markt", "Rue de la Paix")
4. HAUSNUMMER – Hausnummern NUR im Kontext einer Adresse (z.B. "42" in "Hauptstraße 42", "12a"). Einzelne Zahlen ohne Adresskontext sind KEINE Hausnummern!
5. STADT – Städte / Orte (z.B. "Berlin", "Wien", "München", "Graz"). Auch kleinere Orte und Gemeinden.
6. PLZ – Postleitzahlen (z.B. "10115", "A-1010", "8010")
7. LAND – Länder (z.B. "Deutschland", "Österreich", "Germany")
8. KONTONUMMER – Kontonummern, IBANs, BICs, Bankleitzahlen
9. EMAIL – E-Mail-Adressen
10. TELEFON – Telefonnummern, Faxnummern, Mobilnummern (alle Formate)
11. KRYPTO_ADRESSE – Bitcoin-Adressen oder andere Kryptowährungs-Adressen
12. UNTERNEHMEN – Firmennamen (GmbH, AG, Ltd, Inc, SE, OG, KG, e.U., etc.). Auch Kanzleien, Vereine, Stiftungen, Behörden mit spezifischem Namen.
13. GRUNDSTUECK – Grundstücksbezeichnungen, Parzellen, Flurnummern, Grundbucheinträge, EZ-Nummern, KG-Nummern
14. GEBURTSDATUM – Geburtsdaten von Personen (alle Datumsformate)
15. SOZIALVERSICHERUNG – Sozialversicherungsnummern
16. STEUERNUMMER – Steuernummern, UID-Nummern, Finanzamt-Aktenzeichen
17. AUSWEISNUMMER – Reisepass-, Personalausweis-, Führerscheinnummern
18. GELDBETRAG – Geldbeträge und Währungsangaben (z.B. $100, 50€, 1.000 USD, 5.000,00 EUR, £200, CHF 500, ¥10000)
19. UNTERSCHRIFT – Handschriftlich wirkende Texte, Unterschriften, Paraphen, Kürzel, Initialen
20. AKTENZEICHEN – Geschäftszahlen, Aktenzeichen, Referenznummern, Dossiernummern (z.B. "Az. 5 C 123/24", "GZ 2024/0815")

WICHTIGE REGELN:
- GRÜNDLICHKEIT: Gehe den Text Satz für Satz, Wort für Wort durch. Überprüfe JEDEN Eigennamen, JEDE Zahl, JEDE Adresse. Übersehe NICHTS.
- NAMEN SIND PRIORITÄT NR. 1: Jeder Vor- und Nachname MUSS erkannt werden. Prüfe besonders: Briefköpfe, Anreden ("Sehr geehrter Herr ..."), Grußformeln ("Mit freundlichen Grüßen, ..."), Unterschriftszeilen, E-Mail-Header (Von, An, CC), Vertragsparteien, Zeugen, Beteiligte, Bevollmächtigte, Sachbearbeiter, Richter, Anwälte, Notare.
- KONTEXT NUTZEN: Wenn ein Name an einer Stelle im Text vorkommt, prüfe ob derselbe Name oder Teile davon auch an anderen Stellen auftauchen (z.B. "Herr Müller" und später nur "Müller").
- IM ZWEIFEL SCHWÄRZEN: Wenn du dir unsicher bist, ob etwas ein Name, eine Adresse oder andere PII ist – markiere es TROTZDEM. Falsch-positive sind akzeptabel, falsch-negative NICHT.
- Gleiche Entitäten (z.B. derselbe Vorname "Max" an mehreren Stellen) sollen als EINE Entität behandelt werden.
- Gib NUR die Entitäten zurück, die tatsächlich im Text vorkommen.
- Gib die Entitäten EXAKT so zurück, wie sie im Text stehen (gleiche Schreibweise, Groß-/Kleinschreibung).
- Erkenne Entitäten in ALLEN Sprachen (Deutsch, Englisch, Französisch, Türkisch, Arabisch, etc.).
- Ignoriere allgemeine Begriffe die keine konkreten PII sind (z.B. "Straße" allein ohne Straßenname).
- Achte BESONDERS auf Geldbeträge und Währungen: $, €, £, ¥, USD, EUR, GBP, CHF, JPY, BTC und alle anderen Währungen.
- NICHT anonymisieren: Rechtliche Normen, Paragraphen (§, §§), Gesetzesverweise (z.B. "§ 123 BGB", "Art. 5 DSGVO"), Standards und Normen (ISO, DIN, EN, ÖNORM). Diese sind KEINE personenbezogenen Daten.
- NIEMALS anonymisieren: Gliederungsziffern, Nummerierungen und Vertragsstruktur-Elemente! Dazu gehören: "1.", "1.1.", "1.1.1.", "a)", "aa)", "b)", "1.a)", "1.c)", "(1)", "(a)", "I.", "II.", "III.", "IV.", "Nr. 1", "Abs. 1", "Ziff. 3", "lit. a", römische Ziffern, Aufzählungszeichen. Diese Strukturelemente sind KEINE personenbezogenen Daten und dürfen NIEMALS als Entität erkannt werden!
- Erkenne handschriftlich wirkende Texte, Unterschriften, Paraphen, Initialen und Kürzel als UNTERSCHRIFT.

CHECKLISTE VOR DER ANTWORT – Hast du wirklich ALLE gefunden?
- [ ] Alle Vor- und Nachnamen im gesamten Text?
- [ ] Alle Adressen (Straße, Hausnummer, PLZ, Stadt, Land)?
- [ ] Alle Telefonnummern, E-Mails, Kontonummern?
- [ ] Alle Firmennamen?
- [ ] Alle Geldbeträge?
- [ ] Alle Aktenzeichen und Referenznummern?
- [ ] Alle Datumsangaben, die Geburtsdaten sein könnten?
- [ ] Namen in Grußformeln, Unterschriften, Briefköpfen?

Antworte AUSSCHLIESSLICH mit einem JSON-Objekt im folgenden Format, ohne weitere Erklärung:

{
  "entities": [
    {"text": "Max", "category": "VORNAME"},
    {"text": "Mustermann", "category": "NACHNAME"},
    {"text": "Musterstraße", "category": "STRASSE"},
    {"text": "42", "category": "HAUSNUMMER"},
    {"text": "Berlin", "category": "STADT"},
    {"text": "10115", "category": "PLZ"},
    {"text": "DE89370400440532013000", "category": "KONTONUMMER"},
    {"text": "max@example.com", "category": "EMAIL"},
    {"text": "Muster GmbH", "category": "UNTERNEHMEN"},
    {"text": "5.000,00 EUR", "category": "GELDBETRAG"},
    {"text": "5 C 123/24", "category": "AKTENZEICHEN"},
    {"text": "J.M.", "category": "UNTERSCHRIFT"}
  ]
}"""

USER_PROMPT_TEMPLATE = """Analysiere den folgenden Text GRÜNDLICH und finde ALLE personenbezogenen und identifizierenden Daten. Gehe Satz für Satz vor. Übersehe KEINEN einzigen Namen, keine Adresse, keine Nummer.

WICHTIG: Gliederungsziffern (1., 1.1., a), aa), I., II., (1), (a), Nr. 1, Abs. 2, lit. a etc.) und §§-Verweise sind KEINE Entitäten – diese NIEMALS melden!

TEXT:
\"\"\"
{text}
\"\"\"

Antworte NUR mit dem JSON-Objekt. Jeden Namen finden, aber Dokumentstruktur bewahren."""

# ---------------------------------------------------------------------------
# Intensity / scope prompt modifiers
# ---------------------------------------------------------------------------

_INTENSITY_PREFIX = {
    INTENSITY_HARD: (
        "WICHTIGER HINWEIS ZUR INTENSITÄT: Arbeite MAXIMAL GRÜNDLICH. "
        "Im Zweifel schwärzen. Aber: nur ECHTE personenbezogene Daten. "
        "Strukturelemente des Dokuments (Nummerierungen, §§, Gliederungen) "
        "sind KEINE PII und dürfen NIEMALS gemeldet werden.\n\n"
    ),
}

_SCOPE_NAMES_INSTRUCTION = (
    "EINSCHRÄNKUNG DES UMFANGS: Suche nur nach PERSONEN-IDENTIFIZIERENDEN Daten. "
    "Das bedeutet: VORNAME, NACHNAME, STRASSE, HAUSNUMMER, STADT, PLZ, LAND, "
    "EMAIL, TELEFON, UNTERNEHMEN, GEBURTSDATUM, UNTERSCHRIFT, "
    "SOZIALVERSICHERUNG, AUSWEISNUMMER, GRUNDSTUECK. "
    "IGNORIERE: Geldbeträge (GELDBETRAG), Kontonummern (KONTONUMMER), "
    "Krypto-Adressen (KRYPTO_ADRESSE), Steuernummern (STEUERNUMMER), "
    "Aktenzeichen (AKTENZEICHEN) und alle Zahlen/Prozente/Summen.\n\n"
)


def _build_user_prompt(text: str, intensity: str, scope: str) -> str:
    """Build the user prompt with intensity/scope modifiers."""
    prefix = _INTENSITY_PREFIX.get(intensity, "")
    scope_mod = _SCOPE_NAMES_INSTRUCTION if scope == SCOPE_NAMES_ONLY else ""

    base = USER_PROMPT_TEMPLATE.format(text=text)
    if prefix or scope_mod:
        return prefix + scope_mod + base
    return base

# ---------------------------------------------------------------------------
# Prompt for natural replacement generation  (MODE_PSEUDO_NATURAL)
# ---------------------------------------------------------------------------

REPLACEMENT_SYSTEM_PROMPT = """Du bist ein Experte für Datenpseudonymisierung. Deine Aufgabe: Ersetze personenbezogene Daten durch NATÜRLICH KLINGENDE, REALISTISCHE Fake-Daten.

REGELN:
- Vornamen → andere realistische Vornamen (gleiche Sprache/Herkunft wenn erkennbar)
- Nachnamen → andere realistische Nachnamen (gleiche Sprache/Herkunft wenn erkennbar)
- Straßen → andere realistische Straßennamen
- Hausnummern → andere Hausnummern
- Städte → andere Städte im gleichen Land
- PLZ → passende PLZ zur neuen Stadt
- Länder → gleich beibehalten
- Kontonummern/IBANs → andere gültig aussehende Nummern gleicher Länge
- E-Mails → neue E-Mail basierend auf dem neuen Namen
- Telefon → andere Nummer gleichen Formats
- Unternehmen → andere realistische Firmennamen gleicher Art
- Geldbeträge → andere Beträge in ähnlicher Größenordnung
- Geburtsdaten → andere realistische Daten
- Steuernummern/SVN/Ausweisnummern → andere Nummern gleichen Formats
- Aktenzeichen → andere Aktenzeichen gleichen Formats
- Krypto-Adressen → andere Adressen gleichen Formats

WICHTIG:
- KONSISTENZ: Wenn "Max" als Vorname ersetzt wird durch "Thomas", dann ÜBERALL "Thomas".
- Zusammengehörige Daten müssen zueinander passen (E-Mail zum neuen Namen etc.).
- ÄHNLICHE LÄNGE: Die Ersetzung soll möglichst ähnlich viele Zeichen haben wie das Original.
- GLEICHES FORMAT: Die Ersetzung muss das gleiche Format haben (z.B. gleiche Anzahl Ziffern bei Nummern).
- Antworte NUR mit einem JSON-Objekt."""

REPLACEMENT_USER_TEMPLATE = """Erstelle für jede der folgenden Entitäten einen natürlich klingenden Ersatzwert.

Antworte NUR mit einem JSON-Objekt der Form:
{{"replacements": {{"original": "ersatz", ...}}}}

Entitäten:
{entities_json}"""


def _parse_ai_response(response_text: str) -> List[Dict[str, str]]:
    """Parse the JSON response from the AI, handling markdown fences."""
    text = response_text.strip()
    # Strip markdown code fences if present
    fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    data = json.loads(text)
    return data.get("entities", [])


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

MODEL = "gpt-5.2"


def detect_entities_openai(
    api_key: str,
    text: str,
    intensity: str = INTENSITY_HARD,
    scope: str = SCOPE_ALL,
) -> List[Dict[str, str]]:
    """Use OpenAI GPT-5.2 to detect PII entities."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    user_prompt = _build_user_prompt(text, intensity, scope)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_completion_tokens=16384,
    )
    entities = _parse_ai_response(response.choices[0].message.content)

    # Post-filter for person-data scope (belt and suspenders)
    if scope == SCOPE_NAMES_ONLY:
        entities = [e for e in entities if e["category"] in _PERSON_CATEGORIES]

    return entities


def generate_natural_replacements_openai(
    api_key: str, entities: List[Dict[str, str]]
) -> Dict[str, str]:
    """Use OpenAI GPT-5.2 to generate natural-sounding replacement values."""
    # Build concise list (deduplicated, skip signatures)
    items = []
    seen: set = set()
    for ent in entities:
        if ent["text"] not in seen and ent["category"] != "UNTERSCHRIFT":
            items.append({"text": ent["text"], "category": ent["category"]})
            seen.add(ent["text"])

    if not items:
        return {}

    entities_json = json.dumps(items, ensure_ascii=False, indent=2)

    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": REPLACEMENT_SYSTEM_PROMPT},
            {"role": "user", "content": REPLACEMENT_USER_TEMPLATE.format(
                entities_json=entities_json,
            )},
        ],
        temperature=0.7,
        max_completion_tokens=16384,
    )
    text = response.choices[0].message.content.strip()
    fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    data = json.loads(text)
    return data.get("replacements", {})


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

PROVIDERS = {
    "openai": detect_entities_openai,
}

REPLACEMENT_PROVIDERS = {
    "openai": generate_natural_replacements_openai,
}


def _split_text(text: str) -> List[str]:
    """Split *text* into overlapping chunks that fit within AI token limits."""
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP
    return chunks


def _deduplicate_entities(all_entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove duplicate entities (same text + category)."""
    seen = set()
    unique: List[Dict[str, str]] = []
    for ent in all_entities:
        key = (ent["text"], ent["category"])
        if key not in seen:
            seen.add(key)
            unique.append(ent)
    return unique


def detect_entities(
    provider: str,
    api_key: str,
    text: str,
    progress_callback=None,
    intensity: str = INTENSITY_HARD,
    scope: str = SCOPE_ALL,
) -> List[Dict[str, str]]:
    """
    Detect PII entities using the chosen AI provider.

    Automatically splits large texts into chunks and merges results.
    *progress_callback(int)* is called with 0-100 percentage.

    Returns a list of dicts: [{"text": "...", "category": "..."}, ...]
    """
    func = PROVIDERS.get(provider)
    if func is None:
        raise ValueError(f"Unknown provider: {provider}")

    chunks = _split_text(text)
    all_entities: List[Dict[str, str]] = []

    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(int((i / len(chunks)) * 100))
        chunk_entities = func(api_key, chunk, intensity=intensity, scope=scope)
        all_entities.extend(chunk_entities)

    if progress_callback:
        progress_callback(100)

    return _deduplicate_entities(all_entities)


def generate_natural_replacements(
    provider: str,
    api_key: str,
    entities: List[Dict[str, str]],
) -> Dict[str, str]:
    """Generate natural-sounding replacement values using the chosen AI provider.

    Returns a dict mapping original text -> replacement text.
    """
    func = REPLACEMENT_PROVIDERS.get(provider)
    if func is None:
        raise ValueError(f"Unknown provider: {provider}")
    return func(api_key, entities)


def assign_variables(
    entities: List[Dict[str, str]],
    mode: str = MODE_PSEUDO_VARS,
    replacements: Optional[Dict[str, str]] = None,
) -> Dict[str, Tuple[str, str]]:
    """
    Assign labels to detected entities based on the processing mode.

    Modes:
      ``MODE_ANONYMIZE``       – all labels empty (solid black redaction)
      ``MODE_PSEUDO_VARS``     – hexadecimal variable IDs  (A, B, C, …)
      ``MODE_PSEUDO_NATURAL``  – natural-sounding replacement text

    Returns a dict mapping original text -> (label, category).
    Same text always gets the same label.
    """
    mapping: Dict[str, Tuple[str, str]] = {}
    counter = 0xA  # Start at hex A

    for ent in entities:
        txt = ent["text"]
        if txt in mapping:
            continue

        cat = ent["category"]

        # Signatures are always redacted as solid black (no label)
        if cat == "UNTERSCHRIFT":
            mapping[txt] = ("", cat)
            continue

        if mode == MODE_ANONYMIZE:
            mapping[txt] = ("", cat)
        elif mode == MODE_PSEUDO_NATURAL and replacements:
            replacement = replacements.get(txt, f"{counter:X}")
            mapping[txt] = (replacement, cat)
            counter += 1
        else:
            # Default: hex variable IDs
            var_id = f"{counter:X}"
            mapping[txt] = (var_id, cat)
            counter += 1

    return mapping
