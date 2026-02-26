"""
AI Engine – OpenAI GPT-5.2 powered PII entity detection.

Handles large texts by splitting into chunks that fit within AI token limits
and merging results, ensuring consistent variable assignment across chunks.
"""

import json
import re
from typing import Dict, List, Tuple, Optional

# Approximate character limit per chunk.  Most models handle ~120k chars
# comfortably; we stay well below to leave room for the system prompt and
# response.  Overlapping avoids splitting an entity at a boundary.
CHUNK_SIZE = 60_000
CHUNK_OVERLAP = 2_000

# ---------------------------------------------------------------------------
# Prompt that instructs the AI to find all PII entities
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Du bist ein äußerst gründlicher Experte für Datenanonymisierung. Deine Aufgabe ist es, in einem gegebenen Text ALLE personenbezogenen und identifizierenden Daten LÜCKENLOS zu finden.

OBERSTE REGEL: LIEBER ZU VIEL SCHWÄRZEN ALS ZU WENIG. Im Zweifel IMMER als Entität markieren. Es ist viel schlimmer, einen Namen oder eine persönliche Information zu ÜBERSEHEN, als einmal zu viel zu schwärzen.

Du musst folgende Kategorien erkennen – in ALLEN Sprachen, die im Text vorkommen:

1. VORNAME – Vornamen von Personen. Auch: Spitznamen, Rufnamen, abgekürzte Vornamen (z.B. "Max", "M.", "Hans-Peter", "J.", "Dr. Hans"). JEDER Vorname muss erkannt werden, egal ob er am Satzanfang, in einer Aufzählung, in einer Grußformel, in einer Unterschrift, in einem Briefkopf, in einer E-Mail-Signatur oder irgendwo anders steht. Auch einzelne Buchstaben mit Punkt (z.B. "M."), die als Vornamens-Abkürzung verwendet werden.
2. NACHNAME – Nachnamen von Personen. Auch: Doppelnamen (z.B. "Müller-Schmidt"), Namenszusätze (z.B. "von", "van", "de", "zu" als Teil des Namens), Titel+Name-Kombinationen. JEDER Nachname muss erkannt werden, auch wenn er nur einmal vorkommt. Nachnamen in Firmennamen (z.B. "Müller" in "Kanzlei Müller") EBENFALLS erkennen.
3. STRASSE – Straßennamen (z.B. "Hauptstraße", "Bahnhofstr.", "Am Markt", "Rue de la Paix")
4. HAUSNUMMER – Hausnummern (z.B. "42", "12a", "7-9")
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
- NICHT anonymisieren: Rechtliche Normen, Paragraphen (§), Gesetzesverweise (z.B. "§ 123 BGB", "Art. 5 DSGVO"), Standards und Normen (ISO, DIN, EN, ÖNORM). Diese sind KEINE personenbezogenen Daten.
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

USER_PROMPT_TEMPLATE = """Analysiere den folgenden Text EXTREM GRÜNDLICH und finde ALLE personenbezogenen und identifizierenden Daten. Gehe Satz für Satz vor. Übersehe KEINEN einzigen Namen, keine Adresse, keine Nummer. LIEBER ZU VIEL als ZU WENIG schwärzen.

TEXT:
\"\"\"
{text}
\"\"\"

Antworte NUR mit dem JSON-Objekt. Denke daran: Jeden Namen finden, im Zweifel schwärzen."""


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


def detect_entities_openai(api_key: str, text: str) -> List[Dict[str, str]]:
    """Use OpenAI GPT-5.2 to detect PII entities."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
        ],
        temperature=0.0,
        max_completion_tokens=16384,
    )
    return _parse_ai_response(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

PROVIDERS = {
    "openai": detect_entities_openai,
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
        chunk_entities = func(api_key, chunk)
        all_entities.extend(chunk_entities)

    if progress_callback:
        progress_callback(100)

    return _deduplicate_entities(all_entities)


def assign_variables(entities: List[Dict[str, str]]) -> Dict[str, Tuple[str, str]]:
    """
    Assign anonymisation variables to detected entities.

    Returns a dict mapping original text -> (variable_id, category).
    Same text always gets the same variable.  Variables use hexadecimal
    counting starting at A (i.e. A, B, C, D, E, F, 10, 11, …).
    Signature/handwriting entities are redacted without a variable label.
    """
    mapping: Dict[str, Tuple[str, str]] = {}
    counter = 0xA  # Start at hex A
    for ent in entities:
        txt = ent["text"]
        if txt not in mapping:
            if ent["category"] == "UNTERSCHRIFT":
                mapping[txt] = ("", ent["category"])
            else:
                var_id = f"{counter:X}"
                mapping[txt] = (var_id, ent["category"])
                counter += 1
    return mapping
