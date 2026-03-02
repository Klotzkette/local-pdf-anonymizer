# PDF Anonymizer

KI-gestütztes Tool zur automatischen Anonymisierung personenbezogener Daten in PDF-Dokumenten.

## Funktionsweise

1. **Programm starten** – `PDFAnonymizer.exe` (Windows) oder `python src/main.py`
2. **API-Key hinterlegen** – Über ⚙ Einstellungen den Key für OpenAI, Anthropic oder Google Gemini eingeben
3. **PDF laden** – Per Drag & Drop oder über „PDF auswählen"
4. **Speicherort wählen** – Das anonymisierte PDF wird dort abgelegt
5. **Fertig** – Das Tool erkennt automatisch alle PII-Daten und ersetzt sie

## Was wird anonymisiert?

| Kategorie | Beispiel | Variable |
|---|---|---|
| Vornamen | Max → | `VBX01` |
| Nachnamen | Mustermann → | `VBX02` |
| Straßen | Musterstraße → | `VBX03` |
| Hausnummern | 42 → | `VBX04` |
| Städte | Berlin → | `VBX05` |
| Postleitzahlen | 10115 → | `VBX06` |
| Kontonummern / IBAN | DE89 3704 ... → | `VBX07` |
| E-Mail-Adressen | max@example.com → | `VBX08` |
| Krypto-Adressen | 1A1zP1... → | `VBX09` |
| Unternehmensnamen | Muster GmbH → | `VBX10` |
| Grundstücksangaben | Flur 3, Flurstück 42 → | `VBX11` |
| Telefonnummern | +49 30 12345 → | `VBX12` |
| Geburtsdaten | 01.01.1990 → | `VBX13` |
| Steuernummern | DE123456789 → | `VBX14` |

Gleiche Entitäten erhalten immer dieselbe Variable (z. B. „Max" ist überall `VBX01`).

Die anonymisierten Stellen werden **türkis** überdeckt, die Variable wird in weißer Schrift darauf angezeigt.

## Voraussetzungen

- **Das PDF muss bereits Texterkennung (OCR) enthalten.** Gescannte Bilder ohne eingebetteten Text können nicht verarbeitet werden.
- Ein gültiger API-Key für mindestens einen der unterstützten KI-Anbieter:
  - **OpenAI** (ChatGPT) – `sk-...`
  - **Anthropic** (Claude) – `sk-ant-...`
  - **Google Gemini** – `AI...`

## Installation

### Voraussetzungen

- Python 3.10 oder neuer
- pip

### Virtuelle Umgebung erstellen und aktivieren

Es wird empfohlen, das Projekt in einer virtuellen Umgebung (venv) auszuführen, damit die Abhängigkeiten isoliert vom System-Python bleiben.

```bash
# 1. Virtuelle Umgebung erstellen (einmalig)
python -m venv venv

# 2. Virtuelle Umgebung aktivieren
#    Windows (CMD):
venv\Scripts\activate
#    Windows (PowerShell):
venv\Scripts\Activate.ps1
#    macOS / Linux:
source venv/bin/activate

# 3. Abhängigkeiten installieren
pip install -r requirements.txt
```

Nach der Aktivierung erscheint `(venv)` am Anfang der Kommandozeile. Die Umgebung bleibt aktiv, bis man `deactivate` eingibt oder das Terminal schließt.

### Programm starten

```bash
python src/main.py
```

### Als Windows-EXE bauen (optional)

```bash
# Variante 1: über das Build-Skript
build.bat

# Variante 2: direkt mit PyInstaller
pyinstaller build.spec

# Ergebnis: dist\PDFAnonymizer\PDFAnonymizer.exe
```

## Unterstützte KI-Anbieter

| Anbieter | Modell | Kosten |
|---|---|---|
| OpenAI | GPT-4o | nach Verbrauch |
| Anthropic | Claude Sonnet | nach Verbrauch |
| Google | Gemini 2.0 Flash | nach Verbrauch |

## Datenschutz

Der Text des PDFs wird an den gewählten KI-Anbieter gesendet, um die personenbezogenen Daten zu erkennen. Stellen Sie sicher, dass dies mit Ihren Datenschutzanforderungen vereinbar ist.
