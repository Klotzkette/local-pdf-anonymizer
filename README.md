# PDF Anonymizer

KI-gestütztes Tool zur automatischen Anonymisierung personenbezogener Daten in PDF-Dokumenten. Läuft komplett lokal – keine Cloud, kein API-Key, keine Kosten.

## Funktionsweise

1. **Programm starten** – `PDFAnonymizer.exe` (Windows) oder `python src/main.py`
2. **KI-Modell herunterladen** – Beim ersten Start über ⚙ Einstellungen das Modell laden (~6 GB, einmalig)
3. **PDF laden** – Per Drag & Drop oder über „Datei auswählen"
4. **Modus wählen** – Schwärzen oder Pseudonymisieren
5. **Speicherort wählen** – Das anonymisierte PDF wird dort abgelegt
6. **Fertig** – Die lokale KI erkennt automatisch alle personenbezogenen Daten

## Was wird anonymisiert?

| Kategorie | Beispiel |
|---|---|
| Vornamen | Max → geschwärzt / Thomas |
| Nachnamen | Mustermann → geschwärzt / Schmidt |
| Straßen | Musterstraße → geschwärzt / Bahnhofstr. |
| Hausnummern | 42 → geschwärzt / 17 |
| Städte | Berlin → geschwärzt / Hamburg |
| Postleitzahlen | 10115 → geschwärzt / 20095 |
| Kontonummern / IBAN | DE89 3704 ... → geschwärzt |
| E-Mail-Adressen | max@example.com → geschwärzt |
| Unternehmensnamen | Muster GmbH → geschwärzt / Beispiel AG |
| Telefonnummern | +49 30 12345 → geschwärzt |
| Geldbeträge | 5.000,00 EUR → geschwärzt |
| Geburtsdaten | 01.01.1990 → geschwärzt |
| Steuernummern | DE123456789 → geschwärzt |
| Aktenzeichen | 5 C 123/24 → geschwärzt |
| Unterschriften | Handschrift → geschwärzt |

## Modi

- **Schwärzen** – Alle erkannten Daten werden komplett geschwärzt (schwarze Balken)
- **Pseudonymisieren** – Die KI ersetzt erkannte Daten durch natürlich klingende Alternativen

## Voraussetzungen

- Windows 10/11 oder macOS 12+
- Mindestens 16 GB RAM
- ~6 GB freier Speicherplatz für das KI-Modell (Qwen3.5-9B)
- Kein API-Key nötig – alles läuft lokal

## KI-Modell

| | |
|---|---|
| **Modell** | Qwen3.5-9B (9 Milliarden Parameter) |
| **Quelle** | [HuggingFace](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) |
| **Typ** | Sprachmodell, lokal (GGUF Q4_K_M) |
| **Kontext** | 262 000 Tokens |
| **Download** | ~6 GB (einmalig, wird lokal gespeichert) |

## Datenschutz

Alle Daten bleiben auf Ihrem Gerät. Es werden keine Daten an externe Server übertragen. Das KI-Modell läuft vollständig lokal.

## Installation & Build

### Voraussetzungen

- Python 3.10 oder neuer
- pip

### Schritte

```bash
# 1. Abhängigkeiten installieren
pip install -r requirements.txt

# 2. Direkt starten (ohne Build)
python src/main.py

# 3. Oder als EXE bauen
pyinstaller build.spec
```

## Unterstützte Eingabeformate

| Format | Hinweis |
|---|---|
| PDF | Direkt verarbeitet, OCR bei Bedarf |
| DOCX / DOC | Automatisch in PDF konvertiert |
| JPG / JPEG | Automatisch in PDF konvertiert + OCR |
