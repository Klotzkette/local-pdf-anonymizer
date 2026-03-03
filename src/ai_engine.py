"""
AI Engine – Local Qwen3.5-9B powered PII entity detection.

The model is downloaded from HuggingFace on first use and runs entirely
offline afterwards.  No API key required.

Handles large texts by splitting into chunks that fit within AI token limits
and merging results, ensuring consistent variable assignment across chunks.
"""

import json
import re
import threading
from pathlib import Path
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
    "SOZIALVERSICHERUNG", "AUSWEISNUMMER", "GRUNDSTUECK", "STEUERNUMMER",
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

OBERSTE REGEL: Finde ALLE echten personenbezogenen Daten – lieber einmal zu viel als zu wenig. ABER: Dokumentstruktur (Nummerierungen, Paragraphen, Gliederungen) darf NIEMALS als PII gemeldet werden. Das Dokument muss nach der Schwärzung noch lesbar und strukturell intakt sein.

Du musst folgende Kategorien erkennen – in ALLEN Sprachen, die im Text vorkommen:

1. VORNAME – Vornamen von Personen. Auch: Spitznamen, Rufnamen, abgekürzte Vornamen (z.B. "Max", "M.", "Hans-Peter", "J.", "Dr. Hans"). JEDER Vorname muss erkannt werden, egal ob er am Satzanfang, in einer Aufzählung, in einer Grußformel, in einer Unterschrift, in einem Briefkopf, in einer E-Mail-Signatur oder irgendwo anders steht.
2. NACHNAME – Nachnamen von Personen. Auch: Doppelnamen (z.B. "Müller-Schmidt"), Namenszusätze (z.B. "von", "van", "de", "zu" als Teil des Namens). JEDER Nachname muss erkannt werden. Nachnamen in Firmennamen (z.B. "Müller" in "Kanzlei Müller") EBENFALLS.
3. STRASSE – Straßennamen (z.B. "Hauptstraße", "Bahnhofstr.", "Am Markt")
4. HAUSNUMMER – Hausnummern NUR im Kontext einer Adresse (z.B. "42" in "Hauptstraße 42"). Einzelne Zahlen ohne Adresskontext sind KEINE Hausnummern!
5. STADT – Städte / Orte (z.B. "Berlin", "Wien", "München"). Auch kleinere Orte und Gemeinden.
6. PLZ – Postleitzahlen (z.B. "10115", "A-1010", "8010")
7. LAND – Länder (z.B. "Deutschland", "Österreich")
8. KONTONUMMER – Kontonummern, IBANs, BICs, Bankleitzahlen, Depotnummern, Kundennummern bei Banken
9. EMAIL – E-Mail-Adressen (alle Formate)
10. TELEFON – Telefonnummern, Faxnummern, Mobilnummern (alle Formate)
11. KRYPTO_ADRESSE – Bitcoin-, Ethereum- oder andere Kryptowährungs-Adressen und Wallet-IDs
12. UNTERNEHMEN – Firmennamen, Institutsnamen, Banknamen. WICHTIG: Auch spezifische Institutionen wie "Sparkasse Köln-Bonn", "Volksbank Mittelhessen", "Deutsche Bank", "Commerzbank", "Raiffeisenbank", JEDE namentlich genannte Bank, Versicherung, Kanzlei, Behörde, Verein, Stiftung. Auch wenn der Name nur einmal oder beiläufig vorkommt. AUCH Kurzformen wie nur "Sparkasse" oder "Volksbank", wenn sie im Kontext eindeutig eine bestimmte Institution meinen.
13. GRUNDSTUECK – Grundstücksbezeichnungen, Parzellen, Flurnummern, Grundbucheinträge
14. GEBURTSDATUM – Geburtsdaten von Personen (alle Datumsformate)
15. SOZIALVERSICHERUNG – Sozialversicherungsnummern
16. STEUERNUMMER – Steuernummern, UID-Nummern, Finanzamt-Aktenzeichen
17. AUSWEISNUMMER – Reisepass-, Personalausweis-, Führerscheinnummern
18. GELDBETRAG – ALLE Geldbeträge und Währungsangaben. Auch: Gehälter, Mieten, Kaufpreise, Provisionen, Prozentsätze in finanziellem Kontext (z.B. "3,5 %"), Stundensätze, Jahresgehälter, Monatsraten. JEDE Zahl mit Währungssymbol ($, €, £, ¥, CHF) oder Währungscode (USD, EUR, GBP). Auch "brutto", "netto" mit Beträgen.
19. UNTERSCHRIFT – Handschriftlich wirkende Texte, Unterschriften, Paraphen, Kürzel, Initialen
20. AKTENZEICHEN – Geschäftszahlen, Aktenzeichen, Referenznummern, Dossiernummern, Vertragsnummern, Policennummern

WICHTIGE REGELN:
- GRÜNDLICHKEIT: Gehe den Text DREIMAL durch. Prüfe JEDEN Eigennamen, JEDE Zahl, JEDE Adresse, JEDE Institution. ÜBERSEHE NICHTS.
- NAMEN SIND PRIORITÄT NR. 1: Jeder Vor- und Nachname MUSS erkannt werden. Prüfe besonders: Briefköpfe, Anreden, Grußformeln, Unterschriftszeilen, E-Mail-Header, Vertragsparteien, Zeugen, Bevollmächtigte, Sachbearbeiter, Kontoinhaberangaben, Eigentümerfelder.
- INSTITUTSNAMEN SIND PRIORITÄT NR. 2: Jede namentlich genannte Institution muss erkannt werden. Banken (Sparkasse, Volksbank, Deutsche Bank, Commerzbank, etc.), Versicherungen (Allianz, HUK, etc.), Kanzleien, Behörden, Vereine – ALLES was eine konkrete Organisation identifiziert. Auch wenn der Name nur als Kurzform ("die Sparkasse", "bei der Volksbank") auftaucht.
- KONTEXT NUTZEN: Wenn ein Name oder eine Institution an einer Stelle vorkommt, prüfe ob derselbe Name oder Teile davon auch an JEDER anderen Stelle auftauchen – in JEDER Schreibweise: als Kurzform, mit Initialen, umgestellt, in Großbuchstaben, mit/ohne Titel. Beispiele:
  * "Herr Hans Müller" → suche auch nach "Müller", "H. Müller", "MÜLLER", "Müller, Hans"
  * "Sparkasse Köln-Bonn" → suche auch nach "Sparkasse", "SPARKASSE", "SKB"
  * "Dr. Sabine Weber" → suche auch nach "Weber", "S. Weber", "WEBER"
- TECHNISCHE DOKUMENTE – BESONDERE AUFMERKSAMKEIT:
  * KONTOAUSZÜGE: Namen erscheinen in vielen Kontexten – als Kontoinhaber, Auftraggeber, Empfänger, in Überweisungsverwendungszwecken, in Daueraufträgen, als Lastschrift-Mandatsreferenz. JEDER Name in JEDER Buchungszeile muss erkannt werden!
  * GRUNDBUCHAUSZÜGE: Namen erscheinen als Eigentümer, Belastete, Berechtigte, Begünstigte, Gläubiger, Antragsteller, in Eintragungsvermerken, in Abteilungen I-III. Auch Notarnamen, Urkundsbeamte. JEDER Name muss erkannt werden!
  * VERSICHERUNGSDOKUMENTE: Versicherungsnehmer, Begünstigte, Schadensmeldungen – alle Personen und Institutionen.
  * BEHÖRDENBRIEFE: Sachbearbeiter, Antragsteller, Bevollmächtigte, Aktenzeichen mit Namen.
  * GEHALTSABRECHNUNGEN: Arbeitnehmer, Arbeitgeber, Krankenkasse, Finanzamt – alle Namen und Institutionen.
- IM ZWEIFEL SCHWÄRZEN: Wenn du dir unsicher bist – markiere es TROTZDEM. Falsch-positive sind akzeptabel, falsch-negative NICHT.
- Gleiche Entitäten sollen als EINE Entität behandelt werden.
- Gib die Entitäten EXAKT so zurück, wie sie im Text stehen. Wenn derselbe Name in verschiedenen Schreibweisen auftaucht ("Müller" und "MÜLLER"), melde BEIDE Schreibweisen als separate Entitäten.
- Erkenne Entitäten in ALLEN Sprachen.
- NICHT anonymisieren: §§, Gesetzesverweise, Standards (ISO, DIN), generische Begriffe.
- NIEMALS anonymisieren: Gliederungsziffern, Nummerierungen! "1.", "1.1.", "a)", "(1)", "I.", "Nr. 1", "Abs. 1", "lit. a" – diese sind KEINE PII!

CHECKLISTE – GEH DIESE DREIMAL DURCH bevor du antwortest:
- [ ] Alle Vor- und Nachnamen im gesamten Text? (Auch in Briefköpfen, Fußzeilen, Grüßen, Buchungszeilen, Eigentümerfeldern?)
- [ ] Taucht derselbe Name woanders in anderer Schreibweise auf? (Kurzform, Initialen, GROSSBUCHSTABEN, umgestellt, mit Komma?)
- [ ] Alle Firmennamen und Institutsnamen? (Banken, Versicherungen, Kanzleien, Behörden? Auch in Buchungszeilen, Verwendungszwecken?)
- [ ] Alle Adressen (Straße, Hausnummer, PLZ, Stadt, Land)?
- [ ] Alle Telefonnummern, E-Mails, Kontonummern, IBANs?
- [ ] Alle Geldbeträge, Gehälter, Mieten, Prozentsätze?
- [ ] Alle Aktenzeichen, Vertragsnummern, Referenznummern?
- [ ] Alle Geburtsdaten, Steuer- und Sozialversicherungsnummern?
- [ ] Hast du WIRKLICH nichts übersehen? Geh nochmal durch!

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

USER_PROMPT_TEMPLATE = """Analysiere den folgenden Text DREIMAL GRÜNDLICH und finde ALLE personenbezogenen und identifizierenden Daten.

ANLEITUNG:
1. ERSTER DURCHGANG: Gehe Satz für Satz, Zeile für Zeile vor. Markiere alle offensichtlichen Namen, Adressen, Nummern, Institutionen, Beträge. Auch in Tabellen, Buchungszeilen, Verwendungszwecken, Kopf-/Fußzeilen.
2. ZWEITER DURCHGANG: Nimm dir JEDEN bereits gefundenen Namen und JEDE Institution und prüfe ob sie auch woanders vorkommen – in Kurzform, als Initialen ("H. Müller"), umgestellt ("Müller, Hans"), in GROSSBUCHSTABEN ("MÜLLER"), in Fließtext, in Tabellenzeilen. Melde JEDE gefundene Schreibweise als separate Entität. Suche auch nach übersehenen Telefonnummern, E-Mails, IBANs, Geldbeträgen.
3. DRITTER DURCHGANG: Prüfe Briefköpfe, Fußzeilen, Grußformeln, Unterschriftszeilen, Kontoinhaberangaben, Eigentümerfelder, Empfängerfelder, Sachbearbeiterfelder nochmal separat.

ABSOLUT VERBOTEN ALS ENTITÄT: Gliederungsziffern (1., 1.1., a), aa), I., II., (1), (a), Nr. 1, Abs. 2, lit. a etc.), §§-Verweise, Gesetzesnamen (BGB, DSGVO etc.).

TEXT:
\"\"\"
{text}
\"\"\"

Antworte NUR mit dem JSON-Objekt. Jeden Namen und jede Institution in JEDER Schreibweise finden. Dokumentstruktur bewahren."""

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
- Grundstücke → andere Parzellen-/Flurnummern/Grundbucheinträge gleichen Formats
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
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # AI occasionally wraps JSON in extra text – try to extract it
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
        else:
            return []
    entities = data.get("entities", [])
    # Validate structure: each entity must have text + category
    return [e for e in entities if isinstance(e, dict)
            and "text" in e and "category" in e
            and isinstance(e["text"], str) and len(e["text"].strip()) > 0]


# ---------------------------------------------------------------------------
# Local Qwen3.5-9B model – download, load, inference
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_DISPLAY_NAME = "Qwen3.5-9B"

# Module-level model cache (loaded once, reused for every call)
_model = None
_tokenizer = None


def is_model_downloaded() -> bool:
    """Return True if Qwen3.5-9B is available in the local HuggingFace cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache(MODEL_ID, "config.json")
        return result is not None
    except Exception:
        return False


def download_model(progress_callback=None) -> None:
    """Download Qwen3.5-9B from HuggingFace into the local cache.

    *progress_callback(pct: int, msg: str)* is called periodically.
    Pass pct=-1 for indeterminate progress.
    """
    from huggingface_hub import snapshot_download, HfApi

    if progress_callback:
        progress_callback(0, f"Verbinde mit HuggingFace: {MODEL_ID} …")

    # Try to get total expected size for a meaningful progress bar
    total_bytes = 0
    try:
        api = HfApi()
        info = api.model_info(MODEL_ID, files_metadata=True)
        total_bytes = sum(
            (f.size or 0)
            for f in (info.siblings or [])
            if f.rfilename and not f.rfilename.endswith((".gguf", ".ggml"))
        )
    except Exception:
        pass

    # HuggingFace stores the model here
    model_cache_name = "models--" + MODEL_ID.replace("/", "--")
    model_cache_dir = (
        Path.home() / ".cache" / "huggingface" / "hub" / model_cache_name
    )

    download_done = threading.Event()

    def _monitor():
        while not download_done.is_set():
            if total_bytes > 0:
                try:
                    current = sum(
                        f.stat().st_size
                        for f in model_cache_dir.rglob("*")
                        if f.is_file()
                    )
                    pct = min(95, int(current * 100 / total_bytes))
                    msg = (
                        f"Lade herunter: {current / 1e9:.1f} GB"
                        f" / {total_bytes / 1e9:.1f} GB"
                    )
                except Exception:
                    pct = -1
                    msg = "Lade herunter …"
            else:
                pct = -1
                msg = "Lade herunter …"
            if progress_callback:
                progress_callback(pct, msg)
            download_done.wait(3.0)

    monitor_thread = threading.Thread(target=_monitor, daemon=True)
    monitor_thread.start()

    try:
        snapshot_download(
            repo_id=MODEL_ID,
            ignore_patterns=["*.gguf", "*.ggml"],
        )
    finally:
        download_done.set()

    if progress_callback:
        progress_callback(100, "Download abgeschlossen")


def _load_model():
    """Load Qwen3.5-9B into memory (cached globally after first call)."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.bfloat16,
        device_map="auto",
    )
    _model.eval()
    return _model, _tokenizer


def _run_qwen_inference(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_new_tokens: int = 16384,
) -> str:
    """Run a single inference call with the local Qwen model."""
    import torch

    model, tokenizer = _load_model()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Qwen3 supports enable_thinking=False to skip chain-of-thought output
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generate_kwargs: dict = dict(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    if temperature > 0.01:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = 0.9
    else:
        generate_kwargs["do_sample"] = False

    with torch.no_grad():
        generated_ids = model.generate(**generate_kwargs)

    output_ids = generated_ids[0][model_inputs.input_ids.shape[1]:]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def detect_entities_qwen(
    api_key: str,        # unused – kept for API compatibility
    text: str,
    intensity: str = INTENSITY_HARD,
    scope: str = SCOPE_ALL,
) -> List[Dict[str, str]]:
    """Detect PII entities using the local Qwen3.5-9B model."""
    user_prompt = _build_user_prompt(text, intensity, scope)
    response = _run_qwen_inference(SYSTEM_PROMPT, user_prompt, temperature=0.0)
    entities = _parse_ai_response(response)

    if scope == SCOPE_NAMES_ONLY:
        entities = [e for e in entities if e["category"] in _PERSON_CATEGORIES]

    return entities


def generate_natural_replacements_qwen(
    api_key: str,        # unused – kept for API compatibility
    entities: List[Dict[str, str]],
) -> Dict[str, str]:
    """Generate natural-sounding replacements using the local Qwen3.5-9B model."""
    items = []
    seen: set = set()
    for ent in entities:
        if ent["text"] not in seen and ent["category"] != "UNTERSCHRIFT":
            items.append({"text": ent["text"], "category": ent["category"]})
            seen.add(ent["text"])

    if not items:
        return {}

    entities_json = json.dumps(items, ensure_ascii=False, indent=2)
    response = _run_qwen_inference(
        REPLACEMENT_SYSTEM_PROMPT,
        REPLACEMENT_USER_TEMPLATE.format(entities_json=entities_json),
        temperature=0.7,
    )
    text = response.strip()
    fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Try to extract JSON from surrounding text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
        else:
            return {}
    return data.get("replacements", {})


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

PROVIDERS = {
    "qwen": detect_entities_qwen,
}

REPLACEMENT_PROVIDERS = {
    "qwen": generate_natural_replacements_qwen,
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
