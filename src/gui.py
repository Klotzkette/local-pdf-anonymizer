"""
PDF Anonymizer – Modern PyQt6 GUI with drag & drop, API key settings,
and one-click anonymisation workflow.

Design: Dark theme with turquoise accents, smooth animations, clear
visual states (idle → processing → success/error), and a clickable
drop zone that doubles as a progress indicator.
"""

import os
import sys
import json
import traceback
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QProgressBar,
    QMessageBox,
    QDialog,
    QLineEdit,
    QComboBox,
    QGroupBox,
    QFormLayout,
    QFrame,
    QSizePolicy,
    QGraphicsOpacityEffect,
    QSpacerItem,
)
from PyQt6.QtCore import (
    Qt,
    QThread,
    pyqtSignal,
    pyqtProperty,
    QSize,
    QSettings,
    QTimer,
    QPropertyAnimation,
    QEasingCurve,
    QParallelAnimationGroup,
)
from PyQt6.QtGui import (
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QPalette,
    QColor,
    QPainter,
    QPen,
    QBrush,
    QLinearGradient,
    QRadialGradient,
    QMouseEvent,
)

try:
    from ai_engine import (
        detect_entities, assign_variables, generate_natural_replacements,
        MODE_ANONYMIZE, MODE_PSEUDO_VARS, MODE_PSEUDO_NATURAL,
        INTENSITY_HARD,
        SCOPE_NAMES_ONLY, SCOPE_ALL,
    )
    from pdf_processor import (
        extract_text, redact_pdf, get_page_count,
        prepare_input, SUPPORTED_EXTENSIONS,
    )
except ImportError as _imp_err:
    _import_error = _imp_err
    MODE_ANONYMIZE = "anonymize"
    MODE_PSEUDO_VARS = "pseudo_vars"
    MODE_PSEUDO_NATURAL = "pseudo_natural"
    INTENSITY_HARD = "hard"
    SCOPE_NAMES_ONLY = "names_only"
    SCOPE_ALL = "all"
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".jpg", ".jpeg"}
else:
    _import_error = None


# ---------------------------------------------------------------------------
# Settings persistence
# ---------------------------------------------------------------------------

SETTINGS_ORG = "toms_super_simple_pdf_anonymizer"
SETTINGS_APP = "toms_super_simple_pdf_anonymizer"


def _settings() -> QSettings:
    return QSettings(SETTINGS_ORG, SETTINGS_APP)


def save_api_key(provider: str, key: str):
    s = _settings()
    s.setValue(f"api_keys/{provider}", key)


def load_api_key(provider: str) -> str:
    s = _settings()
    return s.value(f"api_keys/{provider}", "")


def save_provider(provider: str):
    s = _settings()
    s.setValue("selected_provider", provider)


def load_provider() -> str:
    s = _settings()
    return s.value("selected_provider", "openai")


def save_output_dir(path: str):
    s = _settings()
    s.setValue("output_dir", path)


def load_output_dir() -> str:
    s = _settings()
    return s.value("output_dir", "")


def save_mode(mode: str):
    s = _settings()
    s.setValue("processing_mode", mode)


def load_mode() -> str:
    s = _settings()
    val = s.value("processing_mode", MODE_ANONYMIZE)
    # Migrate removed mode to default
    if val == MODE_PSEUDO_VARS:
        return MODE_ANONYMIZE
    return val


def save_scope(scope: str):
    s = _settings()
    s.setValue("scope", scope)


def load_scope() -> str:
    s = _settings()
    return s.value("scope", SCOPE_ALL)


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

# Friendlier, warmer colour palette
BLUE            = "#7EB8DA"
BLUE_MID        = "#6EAAD0"
BLUE_DARK       = "#5A9AC4"
BLUE_GLOW       = "#90C8EA"

ORANGE          = "#FFB347"
ORANGE_MID      = "#FFA033"
ORANGE_DARK     = "#F08C20"

BG_DARK         = "#1A1D23"
BG_CARD         = "#22262E"
BG_SURFACE      = "#2A2F38"
BG_HOVER        = "#363C47"

BORDER          = "#3A4050"
BORDER_FOCUS    = "#7EB8DA"

TEXT_PRIMARY    = "#F5F5F0"
TEXT_SECONDARY  = "#A0A8B4"
TEXT_MUTED      = "#7A8290"

SUCCESS         = "#6FCF7F"
ERROR           = "#F07070"
WARNING         = "#E0B040"


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------

STYLESHEET = f"""
/* ── Global ────────────────────────────────────────────── */
QMainWindow {{
    background-color: {BG_DARK};
}}
QWidget#centralWidget {{
    background-color: {BG_DARK};
}}
QLabel {{
    color: {TEXT_SECONDARY};
    font-size: 13px;
    background: transparent;
}}

/* ── Typography ────────────────────────────────────────── */
QLabel#titleLabel {{
    color: {TEXT_PRIMARY};
    font-size: 24px;
    font-weight: 700;
    letter-spacing: -0.5px;
}}
QLabel#titleAccent {{
    color: {ORANGE};
    font-size: 24px;
    font-weight: 700;
}}
QLabel#subtitleLabel {{
    color: {TEXT_SECONDARY};
    font-size: 12px;
    line-height: 1.5;
}}
QLabel#dropIcon {{
    font-size: 56px;
    background: transparent;
}}
QLabel#dropLabel {{
    color: {TEXT_PRIMARY};
    font-size: 15px;
    font-weight: 600;
}}
QLabel#dropHint {{
    color: {TEXT_MUTED};
    font-size: 11px;
}}
QLabel#fileLabel {{
    color: {BLUE};
    font-size: 13px;
    font-weight: 600;
    padding: 4px 0px;
}}
QLabel#statusLabel {{
    color: {TEXT_SECONDARY};
    font-size: 12px;
}}
QLabel#stepLabel {{
    color: {ORANGE};
    font-size: 11px;
    font-weight: 600;
}}
QLabel#providerPill {{
    color: {BLUE};
    background-color: rgba(91, 164, 207, 0.08);
    border: 1px solid rgba(91, 164, 207, 0.25);
    border-radius: 10px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 600;
}}
QLabel#successLabel {{
    color: {SUCCESS};
    font-size: 13px;
    font-weight: 600;
}}
QLabel#errorLabel {{
    color: {ERROR};
    font-size: 13px;
    font-weight: 600;
}}

/* ── Buttons ───────────────────────────────────────────── */
QPushButton {{
    background-color: {BLUE_DARK};
    color: {TEXT_PRIMARY};
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-size: 13px;
    font-weight: 600;
}}
QPushButton:hover {{
    background-color: {BLUE_MID};
}}
QPushButton:pressed {{
    background-color: {ORANGE};
    color: {BG_DARK};
}}
QPushButton:disabled {{
    background-color: {BG_SURFACE};
    color: {TEXT_MUTED};
}}
QPushButton#settingsBtn {{
    background-color: transparent;
    color: {TEXT_SECONDARY};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 500;
}}
QPushButton#settingsBtn:hover {{
    color: {TEXT_PRIMARY};
    border-color: {TEXT_SECONDARY};
    background-color: {BG_SURFACE};
}}
QPushButton#selectBtn {{
    background-color: transparent;
    color: {BLUE};
    border: 1px solid {BLUE_DARK};
    border-radius: 8px;
    padding: 8px 20px;
    font-size: 13px;
}}
QPushButton#selectBtn:hover {{
    background-color: rgba(91, 164, 207, 0.1);
    border-color: {BLUE};
}}
QPushButton#selectBtn:disabled {{
    color: {TEXT_MUTED};
    border-color: {BORDER};
    background-color: transparent;
}}
QPushButton#openFolderBtn {{
    background-color: transparent;
    color: {SUCCESS};
    border: 1px solid rgba(63, 185, 80, 0.4);
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 12px;
}}
QPushButton#openFolderBtn:hover {{
    background-color: rgba(63, 185, 80, 0.1);
    border-color: {SUCCESS};
}}

/* ── Progress ──────────────────────────────────────────── */
QProgressBar {{
    border: none;
    border-radius: 4px;
    text-align: center;
    color: {TEXT_PRIMARY};
    background-color: {BG_SURFACE};
    max-height: 8px;
    min-height: 8px;
    font-size: 0px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {BLUE_DARK}, stop:1 {ORANGE});
    border-radius: 4px;
}}

/* ── Inputs ────────────────────────────────────────────── */
QComboBox {{
    background-color: {BG_SURFACE};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 13px;
}}
QComboBox:focus {{
    border-color: {BLUE_DARK};
}}
QComboBox QAbstractItemView {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    selection-background-color: {BG_SURFACE};
    border: 1px solid {BORDER};
    outline: none;
}}
QComboBox::drop-down {{
    border: none;
    padding-right: 8px;
}}
QLineEdit {{
    background-color: {BG_SURFACE};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 13px;
    selection-background-color: {BLUE_DARK};
}}
QLineEdit:focus {{
    border-color: {BLUE_DARK};
}}
QLineEdit[valid="true"] {{
    border-color: {SUCCESS};
}}

/* ── Groups ────────────────────────────────────────────── */
QGroupBox {{
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 14px;
    padding: 20px 16px 12px 16px;
    font-size: 13px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 8px;
    color: {TEXT_PRIMARY};
}}

/* ── Dialog ────────────────────────────────────────────── */
QDialog {{
    background-color: {BG_DARK};
}}

/* ── Status bar ────────────────────────────────────────── */
QStatusBar {{
    background-color: {BG_CARD};
    color: {TEXT_MUTED};
    font-size: 11px;
    border-top: 1px solid {BORDER};
    padding: 2px 8px;
}}

/* ── Drop zone ─────────────────────────────────────────── */
QFrame#dropZone {{
    background-color: {BG_CARD};
    border: 2px dashed {BORDER};
    border-radius: 16px;
}}
QFrame#dropZone:hover {{
    border-color: {BLUE_DARK};
    background-color: rgba(91, 164, 207, 0.03);
}}
QFrame#dropZone[dragOver="true"] {{
    border-color: {ORANGE};
    border-style: solid;
    background-color: rgba(255, 159, 67, 0.06);
}}
QFrame#dropZone[processing="true"] {{
    border-color: {BLUE_DARK};
    border-style: solid;
}}
QFrame#dropZone[success="true"] {{
    border-color: {SUCCESS};
    border-style: solid;
    background-color: rgba(63, 185, 80, 0.04);
}}
QFrame#dropZone[error="true"] {{
    border-color: {ERROR};
    border-style: solid;
    background-color: rgba(248, 81, 73, 0.04);
}}

/* ── Scrollbar (thin) ──────────────────────────────────── */
QScrollBar:vertical {{
    background-color: transparent;
    width: 6px;
}}
QScrollBar::handle:vertical {{
    background-color: {BG_HOVER};
    border-radius: 3px;
    min-height: 30px;
}}
"""


# ---------------------------------------------------------------------------
# Animated drop zone with visual states
# ---------------------------------------------------------------------------

_ACCEPTED_EXTENSIONS = tuple(SUPPORTED_EXTENSIONS)  # (".pdf", ".docx", ...)


class DropZone(QFrame):
    """Drop zone that accepts PDF, DOCX, DOC, JPG, JPEG files."""
    file_dropped = pyqtSignal(str)
    clicked = pyqtSignal()

    STATE_IDLE = "idle"
    STATE_PROCESSING = "processing"
    STATE_SUCCESS = "success"
    STATE_ERROR = "error"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(240)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._state = self.STATE_IDLE

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(6)

        # Icon
        self.icon_label = QLabel()
        self.icon_label.setObjectName("dropIcon")
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.icon_label)

        # Primary label
        self.primary_label = QLabel()
        self.primary_label.setObjectName("dropLabel")
        self.primary_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.primary_label)

        # Secondary label
        self.secondary_label = QLabel()
        self.secondary_label.setObjectName("dropHint")
        self.secondary_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.secondary_label)

        # Step indicator (shown during processing)
        self.step_label = QLabel()
        self.step_label.setObjectName("stepLabel")
        self.step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.step_label.setVisible(False)
        layout.addWidget(self.step_label)

        # Progress bar (inside the drop zone)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setMaximumWidth(320)
        layout.addWidget(self.progress_bar, alignment=Qt.AlignmentFlag.AlignCenter)

        self.set_state(self.STATE_IDLE)

    def set_state(self, state: str, detail: str = ""):
        self._state = state
        # Reset all dynamic properties
        for prop in ("dragOver", "processing", "success", "error"):
            self.setProperty(prop, False)

        if state == self.STATE_IDLE:
            self.icon_label.setText("\U0001F4C4")  # document emoji
            self.primary_label.setText("Datei hier ablegen")
            self.secondary_label.setText("PDF, Word, JPG – oder klicken zum Auswählen")
            self.secondary_label.setVisible(True)
            self.step_label.setVisible(False)
            self.progress_bar.setVisible(False)
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setAcceptDrops(True)

        elif state == self.STATE_PROCESSING:
            self.setProperty("processing", True)
            self.icon_label.setText("\U0001F50D")  # magnifying glass
            self.primary_label.setText("Wird verarbeitet …")
            self.secondary_label.setVisible(False)
            self.step_label.setVisible(True)
            self.step_label.setText(detail or "Initialisiere …")
            self.progress_bar.setVisible(True)
            self.setCursor(Qt.CursorShape.WaitCursor)
            self.setAcceptDrops(False)

        elif state == self.STATE_SUCCESS:
            self.setProperty("success", True)
            self.icon_label.setText("\u2705")  # check mark
            self.primary_label.setText("Anonymisierung abgeschlossen")
            self.primary_label.setStyleSheet(f"color: {SUCCESS};")
            self.secondary_label.setText(detail or "")
            self.secondary_label.setVisible(bool(detail))
            self.step_label.setVisible(False)
            self.progress_bar.setVisible(False)
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setAcceptDrops(True)

        elif state == self.STATE_ERROR:
            self.setProperty("error", True)
            self.icon_label.setText("\u274C")  # cross mark
            self.primary_label.setText("Fehler aufgetreten")
            self.primary_label.setStyleSheet(f"color: {ERROR};")
            self.secondary_label.setText("Klicken oder neue Datei ablegen, um es erneut zu versuchen")
            self.secondary_label.setVisible(True)
            self.step_label.setVisible(False)
            self.progress_bar.setVisible(False)
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setAcceptDrops(True)

        # Re-apply style to pick up property changes
        if state != self.STATE_PROCESSING:
            self.primary_label.setStyleSheet("")  # reset inline override
        if state == self.STATE_SUCCESS:
            self.primary_label.setStyleSheet(f"color: {SUCCESS};")
        elif state == self.STATE_ERROR:
            self.primary_label.setStyleSheet(f"color: {ERROR};")

        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def set_progress(self, value: int):
        self.progress_bar.setValue(value)

    def set_step(self, text: str):
        self.step_label.setText(text)

    # -- Events --

    def mousePressEvent(self, event: QMouseEvent):
        if self._state != self.STATE_PROCESSING:
            self.clicked.emit()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith(_ACCEPTED_EXTENSIONS):
                    event.acceptProposedAction()
                    self.setProperty("dragOver", True)
                    self.style().unpolish(self)
                    self.style().polish(self)
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        self.setProperty("dragOver", False)
        self.style().unpolish(self)
        self.style().polish(self)

    def dropEvent(self, event: QDropEvent):
        self.setProperty("dragOver", False)
        self.style().unpolish(self)
        self.style().polish(self)
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(_ACCEPTED_EXTENSIONS):
                self.file_dropped.emit(path)
                return


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

class AnonymizeWorker(QThread):
    progress = pyqtSignal(int)           # 0-100
    status = pyqtSignal(str)             # status text
    step = pyqtSignal(str)               # step indicator  (1/5, 2/5 …)
    entity_count = pyqtSignal(int)       # number of entities found
    finished_ok = pyqtSignal(str)        # output path
    finished_err = pyqtSignal(str)       # error message

    def __init__(
        self,
        input_path: str,
        output_path: str,
        provider: str,
        api_key: str,
        mode: str = MODE_ANONYMIZE,
        scope: str = SCOPE_ALL,
    ):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.provider = provider
        self.api_key = api_key
        self.mode = mode
        self.scope = scope
        self._temp_pdf: str | None = None

    def run(self):
        try:
            # Step 1 – prepare input (convert / OCR if needed)
            self.step.emit("Schritt 1/5  –  Datei vorbereiten")
            self.status.emit("Eingabedatei wird vorbereitet …")
            self.progress.emit(2)

            pdf_path = prepare_input(
                self.input_path,
                status_callback=lambda msg: self.status.emit(msg),
            )
            if pdf_path != self.input_path:
                self._temp_pdf = pdf_path  # remember for cleanup

            # Step 2 – extract text
            self.step.emit("Schritt 2/5  –  Text extrahieren")
            self.status.emit("Text wird aus dem PDF extrahiert …")
            self.progress.emit(8)
            text = extract_text(pdf_path)

            # Step 3 – AI entity detection
            self.step.emit("Schritt 3/5  –  KI-Analyse")
            self.status.emit("KI analysiert den Text …")
            self.progress.emit(12)

            def _ai_progress(pct):
                self.progress.emit(12 + int(pct * 0.28))

            entities = detect_entities(
                self.provider, self.api_key, text,
                progress_callback=_ai_progress,
                intensity=INTENSITY_HARD,
                scope=self.scope,
            )

            if not entities:
                self.entity_count.emit(0)
                self.status.emit("Keine personenbezogenen Daten gefunden.")
                self.progress.emit(100)
                import shutil
                shutil.copy2(pdf_path, self.output_path)
                self._cleanup_temp()
                self.finished_ok.emit(self.output_path)
                return

            self.entity_count.emit(len(entities))

            # Step 4 – assign labels (variables / natural replacements)
            replacements = None
            if self.mode == MODE_PSEUDO_NATURAL:
                self.step.emit("Schritt 4/5  –  Natürliche Ersetzungen generieren")
                self.status.emit(f"{len(entities)} Entitäten erkannt – generiere Ersetzungen …")
                self.progress.emit(42)
                replacements = generate_natural_replacements(
                    self.provider, self.api_key, entities,
                )
            else:
                self.step.emit("Schritt 4/5  –  Variablen zuweisen")
                self.status.emit(f"{len(entities)} Entitäten erkannt …")
                self.progress.emit(42)

            entity_map = assign_variables(entities, mode=self.mode, replacements=replacements)

            # Step 5 – redact PDF
            mode_label = {
                MODE_ANONYMIZE: "anonymisieren",
                MODE_PSEUDO_VARS: "pseudonymisieren",
                MODE_PSEUDO_NATURAL: "pseudonymisieren",
            }.get(self.mode, "verarbeiten")
            self.step.emit(f"Schritt 5/5  –  PDF {mode_label}")
            self.status.emit("PDF wird geschrieben …")

            def _pdf_progress(pct):
                self.progress.emit(45 + int(pct * 0.50))

            redact_pdf(
                pdf_path, self.output_path, entity_map,
                mode=self.mode, progress_callback=_pdf_progress,
                api_key=self.api_key,
            )

            self._cleanup_temp()
            self.progress.emit(100)
            self.finished_ok.emit(self.output_path)

        except Exception as e:
            self._cleanup_temp()
            self.finished_err.emit(f"{e}\n\n{traceback.format_exc()}")

    def _cleanup_temp(self):
        """Remove temporary PDF created during conversion/OCR."""
        if self._temp_pdf:
            try:
                os.unlink(self._temp_pdf)
            except OSError:
                pass
            self._temp_pdf = None


# ---------------------------------------------------------------------------
# Settings dialog (polished)
# ---------------------------------------------------------------------------

PROVIDER_NAMES = {
    "openai":    "OpenAI (GPT-5.2)",
}
PROVIDER_KEYS = ["openai"]
PROVIDER_PLACEHOLDERS = {
    "openai":    "sk-...",
}


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Einstellungen")
        self.setMinimumWidth(500)
        self.setStyleSheet(STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setSpacing(18)
        layout.setContentsMargins(28, 28, 28, 24)

        # -- Header --
        header = QLabel("Einstellungen")
        header.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 18px; font-weight: 700;")
        layout.addWidget(header)
        desc = QLabel("Hinterlegen Sie Ihren OpenAI API-Key. Es wird GPT-5.2 verwendet.")
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
        layout.addWidget(desc)

        layout.addSpacing(4)

        # -- Model info --
        model_label = QLabel("Modell:  GPT-5.2")
        model_label.setStyleSheet(
            f"color: {BLUE}; font-size: 13px; font-weight: 600; "
            f"background-color: rgba(91, 164, 207, 0.08); "
            f"border: 1px solid rgba(91, 164, 207, 0.25); "
            f"border-radius: 8px; padding: 8px 14px;"
        )
        layout.addWidget(model_label)

        layout.addSpacing(4)

        # -- API key --
        keys_group = QGroupBox("OpenAI API-Key")
        keys_layout = QFormLayout(keys_group)
        keys_layout.setSpacing(10)

        self.key_field = QLineEdit(load_api_key("openai"))
        self.key_field.setPlaceholderText("sk-...")
        self.key_field.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_field.setMinimumHeight(36)
        self.key_field.textChanged.connect(self._on_key_changed)
        keys_layout.addRow("API-Key:", self.key_field)

        layout.addWidget(keys_group)
        layout.addStretch()

        # -- Buttons --
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        btn_layout.addStretch()

        cancel_btn = QPushButton("Abbrechen")
        cancel_btn.setObjectName("selectBtn")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        save_btn = QPushButton("Speichern")
        save_btn.setMinimumWidth(120)
        save_btn.clicked.connect(self.save_and_close)
        btn_layout.addWidget(save_btn)

        layout.addLayout(btn_layout)

    def _on_key_changed(self):
        """Give a green border hint when the key field has content."""
        has_value = bool(self.key_field.text().strip())
        self.key_field.setProperty("valid", has_value)
        self.key_field.style().unpolish(self.key_field)
        self.key_field.style().polish(self.key_field)

    def save_and_close(self):
        save_api_key("openai", self.key_field.text().strip())
        save_provider("openai")
        self.accept()


# ---------------------------------------------------------------------------
# Mode selection dialog  (shown after file selection)
# ---------------------------------------------------------------------------

_MODE_OPTIONS = [
    (
        MODE_ANONYMIZE,
        "Schwärzen",
        "Alle erkannten Daten werden komplett\n"
        "geschwärzt – nichts bleibt lesbar.",
    ),
    (
        MODE_PSEUDO_NATURAL,
        "Pseudonymisieren",
        "KI ersetzt Namen, Adressen, Nummern etc.\n"
        "durch natürlich klingende Alternativen.",
    ),
]

_SCOPE_OPTIONS = [
    (SCOPE_NAMES_ONLY, "Personenbezogene Daten",
     "Namen, Adressen, Städte, E-Mail, Telefon\nund weitere personen-identifizierende Infos"),
    (SCOPE_ALL, "Zusätzlich Nummern & Beträge",
     "Wie oben, plus Geldbeträge, Summen,\nProzente, IBANs, Aktenzeichen etc."),
]


class _ChipGroup(QFrame):
    """A row of clickable chip buttons for option selection."""

    selection_changed = pyqtSignal(str)

    def __init__(self, options: list, saved_value: str, parent=None):
        super().__init__(parent)
        self._selected = saved_value
        self._chips: dict = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        for key, label, tooltip in options:
            chip = QPushButton(label)
            chip.setToolTip(tooltip)
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.setFixedHeight(32)
            chip.clicked.connect(lambda checked, k=key: self._on_click(k))
            self._chips[key] = chip
            layout.addWidget(chip)

        layout.addStretch()
        self._update_styles()

    def _on_click(self, key: str):
        self._selected = key
        self._update_styles()
        self.selection_changed.emit(key)

    def _update_styles(self):
        for key, chip in self._chips.items():
            if key == self._selected:
                chip.setStyleSheet(
                    f"QPushButton {{ background-color: {ORANGE}; color: {BG_DARK}; "
                    f"border: none; border-radius: 6px; padding: 4px 14px; "
                    f"font-size: 12px; font-weight: 600; }}"
                )
            else:
                chip.setStyleSheet(
                    f"QPushButton {{ background-color: {BG_SURFACE}; color: {TEXT_SECONDARY}; "
                    f"border: 1px solid {BORDER}; border-radius: 6px; padding: 4px 14px; "
                    f"font-size: 12px; font-weight: 500; }}"
                    f"QPushButton:hover {{ border-color: {ORANGE}; color: {TEXT_PRIMARY}; }}"
                )

    @property
    def selected(self) -> str:
        return self._selected


class ModeSelectionDialog(QDialog):
    """Dialog shown after file selection to choose mode and scope."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Verarbeitungsoptionen")
        self.setFixedWidth(500)
        self.setStyleSheet(STYLESHEET)
        self.selected_mode: str | None = None
        self.selected_scope: str = load_scope()

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(28, 24, 28, 20)

        # Header
        header = QLabel("Verarbeitungsoptionen")
        header.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 16px; font-weight: 700;"
        )
        layout.addWidget(header)
        layout.addSpacing(2)

        # -- Scope section --
        scope_label = QLabel("Umfang")
        scope_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px; font-weight: 600; "
            f"text-transform: uppercase; letter-spacing: 1px;"
        )
        layout.addWidget(scope_label)

        self._scope_chips = _ChipGroup(
            _SCOPE_OPTIONS, self.selected_scope
        )
        self._scope_chips.selection_changed.connect(self._on_scope)
        layout.addWidget(self._scope_chips)

        layout.addSpacing(8)

        # -- Divider --
        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background-color: {BORDER};")
        layout.addWidget(divider)

        layout.addSpacing(4)

        # -- Mode section --
        mode_label = QLabel("Modus  (zum Starten anklicken)")
        mode_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px; font-weight: 600; "
            f"text-transform: uppercase; letter-spacing: 1px;"
        )
        layout.addWidget(mode_label)
        layout.addSpacing(2)

        saved_mode = load_mode()

        for mode_key, title, desc in _MODE_OPTIONS:
            is_saved = mode_key == saved_mode
            card = QFrame()
            card.setCursor(Qt.CursorShape.PointingHandCursor)
            border_col = ORANGE if is_saved else BORDER
            card.setStyleSheet(f"""
                QFrame {{
                    background-color: {BG_SURFACE};
                    border: {"2" if is_saved else "1"}px solid {border_col};
                    border-radius: 10px;
                    padding: 12px 14px;
                }}
                QFrame:hover {{
                    border-color: {ORANGE};
                    background-color: rgba(255, 179, 71, 0.06);
                }}
            """)

            card_layout = QVBoxLayout(card)
            card_layout.setSpacing(3)
            card_layout.setContentsMargins(0, 0, 0, 0)

            title_label = QLabel(title)
            title_label.setStyleSheet(
                f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600; "
                f"border: none; background: transparent;"
            )
            card_layout.addWidget(title_label)

            desc_label = QLabel(desc)
            desc_label.setStyleSheet(
                f"color: {TEXT_SECONDARY}; font-size: 11px; "
                f"border: none; background: transparent;"
            )
            desc_label.setWordWrap(True)
            card_layout.addWidget(desc_label)

            card.mousePressEvent = lambda event, m=mode_key: self._select(m)
            layout.addWidget(card)

        layout.addSpacing(4)

        # Cancel button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Abbrechen")
        cancel_btn.setObjectName("selectBtn")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def _on_scope(self, value: str):
        self.selected_scope = value
        save_scope(value)

    def _select(self, mode: str):
        self.selected_mode = mode
        save_mode(mode)
        self.accept()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tom's Super Simple PDF Anonymizer")
        self.setMinimumSize(640, 560)
        self.resize(720, 620)
        self.setStyleSheet(STYLESHEET)

        self.worker = None
        self.current_pdf: str | None = None
        self._last_output: str | None = None
        self._entity_count = 0
        self._selected_mode: str = MODE_ANONYMIZE

        # Central widget
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(32, 24, 32, 16)
        main_layout.setSpacing(0)

        # ── Header ──
        header = QHBoxLayout()
        header.setSpacing(8)

        title_layout = QHBoxLayout()
        title_layout.setSpacing(0)
        t1 = QLabel("Tom's Super Simple ")
        t1.setObjectName("titleLabel")
        title_layout.addWidget(t1)
        t2 = QLabel("PDF Anonymizer")
        t2.setObjectName("titleAccent")
        title_layout.addWidget(t2)
        title_layout.addStretch()
        header.addLayout(title_layout, stretch=1)

        # Provider pill
        self.provider_pill = QLabel()
        self.provider_pill.setObjectName("providerPill")
        self._update_provider_pill()
        header.addWidget(self.provider_pill, alignment=Qt.AlignmentFlag.AlignVCenter)

        header.addSpacing(8)

        settings_btn = QPushButton("Einstellungen")
        settings_btn.setObjectName("settingsBtn")
        settings_btn.clicked.connect(self.open_settings)
        header.addWidget(settings_btn, alignment=Qt.AlignmentFlag.AlignVCenter)

        main_layout.addLayout(header)
        main_layout.addSpacing(4)

        # Subtitle
        subtitle = QLabel(
            "Automatische KI-gestützte Anonymisierung personenbezogener Daten.  "
            "Unterstützt: PDF, Word (DOCX), JPG – OCR wird bei Bedarf automatisch durchgeführt."
        )
        subtitle.setObjectName("subtitleLabel")
        subtitle.setWordWrap(True)
        main_layout.addWidget(subtitle)
        main_layout.addSpacing(16)

        # ── Drop zone ──
        self.drop_zone = DropZone()
        self.drop_zone.file_dropped.connect(self.on_file_selected)
        self.drop_zone.clicked.connect(self.browse_pdf)
        main_layout.addWidget(self.drop_zone, stretch=1)
        main_layout.addSpacing(12)

        # ── Bottom bar (file info + buttons) ──
        bottom = QHBoxLayout()
        bottom.setSpacing(10)

        self.file_label = QLabel("Keine Datei ausgewählt")
        self.file_label.setObjectName("fileLabel")
        self.file_label.setWordWrap(True)
        bottom.addWidget(self.file_label, stretch=1)

        self.open_folder_btn = QPushButton("Ordner öffnen")
        self.open_folder_btn.setObjectName("openFolderBtn")
        self.open_folder_btn.setVisible(False)
        self.open_folder_btn.clicked.connect(self._open_output_folder)
        bottom.addWidget(self.open_folder_btn)

        self.select_btn = QPushButton("Datei auswählen")
        self.select_btn.setObjectName("selectBtn")
        self.select_btn.clicked.connect(self.browse_pdf)
        bottom.addWidget(self.select_btn)

        main_layout.addLayout(bottom)
        main_layout.addSpacing(4)

        # ── Status bar ──
        self.statusBar().showMessage("Bereit")
        self._update_statusbar_idle()

    # -- Helpers --

    def _update_provider_pill(self):
        has_key = bool(load_api_key("openai"))
        if has_key:
            self.provider_pill.setText("GPT-5.2")
        else:
            self.provider_pill.setText("GPT-5.2  (kein Key)")
            self.provider_pill.setStyleSheet(
                f"color: {WARNING}; background-color: rgba(210, 153, 34, 0.08); "
                f"border: 1px solid rgba(210, 153, 34, 0.25); "
                f"border-radius: 10px; padding: 3px 10px; font-size: 11px; font-weight: 600;"
            )
            return
        self.provider_pill.setStyleSheet("")  # reset to default from stylesheet

    def _update_statusbar_idle(self):
        prov = load_provider()
        has_key = bool(load_api_key(prov))
        if has_key:
            self.statusBar().showMessage("Bereit  –  PDF ablegen oder auswählen")
        else:
            self.statusBar().showMessage("Bitte zuerst einen API-Key in den Einstellungen hinterlegen")

    def _current_mode(self) -> str:
        return self._selected_mode

    def _set_processing(self, active: bool):
        """Lock/unlock UI during processing."""
        self.select_btn.setEnabled(not active)

    def _open_output_folder(self):
        if self._last_output:
            folder = os.path.dirname(self._last_output)
            if sys.platform == "win32":
                os.startfile(folder)
            elif sys.platform == "darwin":
                os.system(f'open "{folder}"')
            else:
                os.system(f'xdg-open "{folder}"')

    # -- Slots --

    def open_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec():
            self._update_provider_pill()
            self._update_statusbar_idle()

    def browse_pdf(self):
        if self.worker and self.worker.isRunning():
            return
        file_filter = (
            "Alle unterstützten Dateien (*.pdf *.docx *.doc *.jpg *.jpeg);;"
            "PDF-Dateien (*.pdf);;"
            "Word-Dokumente (*.docx *.doc);;"
            "Bilder (*.jpg *.jpeg)"
        )
        path, _ = QFileDialog.getOpenFileName(
            self, "Datei auswählen", "", file_filter,
        )
        if path:
            self.on_file_selected(path)

    def on_file_selected(self, path: str):
        if self.worker and self.worker.isRunning():
            return
        self.current_pdf = path
        self._last_output = None
        self.open_folder_btn.setVisible(False)
        name = os.path.basename(path)
        self.file_label.setText(name)
        self.statusBar().showMessage(f"Geladen: {name}")
        self.start_anonymization()

    # Class-level counter so each file in a session gets a unique number
    _file_counter = 0

    @classmethod
    def _anonymized_filename(cls, mode: str) -> str:
        """Generate an output filename: date + suffix + running number."""
        from datetime import date
        cls._file_counter += 1
        today = date.today().strftime("%Y%m%d")
        suffix = "Geschwärzt" if mode == MODE_ANONYMIZE else "Pseudonymisiert"
        return f"{today}_Dokument_{suffix}_{cls._file_counter:03d}.pdf"

    def start_anonymization(self):
        if not self.current_pdf:
            return

        # Check API key
        provider = load_provider()
        api_key = load_api_key(provider)
        if not api_key:
            QMessageBox.warning(
                self,
                "Kein API-Key",
                f"Bitte hinterlegen Sie zuerst einen API-Key für\n"
                f"{PROVIDER_NAMES.get(provider, provider)} in den Einstellungen.",
            )
            self.open_settings()
            api_key = load_api_key(load_provider())
            if not api_key:
                return

        # Show mode selection dialog
        mode_dlg = ModeSelectionDialog(self)
        if not mode_dlg.exec():
            self.drop_zone.set_state(DropZone.STATE_IDLE)
            return
        mode = mode_dlg.selected_mode
        scope = mode_dlg.selected_scope
        self._selected_mode = mode

        # Ask for output location (anonymized filename)
        default_dir = load_output_dir() or os.path.dirname(self.current_pdf)
        default_name = self._anonymized_filename(mode)
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Anonymisiertes PDF speichern unter",
            os.path.join(default_dir, default_name),
            "PDF-Dateien (*.pdf)",
        )
        if not output_path:
            self.drop_zone.set_state(DropZone.STATE_IDLE)
            return

        save_output_dir(os.path.dirname(output_path))

        # Enter processing state
        self._set_processing(True)
        self.drop_zone.set_state(DropZone.STATE_PROCESSING, "Initialisiere …")
        self._entity_count = 0

        # Launch worker
        self.worker = AnonymizeWorker(
            self.current_pdf, output_path, provider, api_key,
            mode=mode, scope=scope,
        )
        self.worker.progress.connect(self.drop_zone.set_progress)
        self.worker.step.connect(self.drop_zone.set_step)
        self.worker.status.connect(lambda s: self.statusBar().showMessage(s))
        self.worker.entity_count.connect(self._on_entity_count)
        self.worker.finished_ok.connect(self.on_success)
        self.worker.finished_err.connect(self.on_error)
        self.worker.start()

    def _on_entity_count(self, count: int):
        self._entity_count = count

    def on_success(self, output_path: str):
        self._set_processing(False)
        self._last_output = output_path

        out_name = os.path.basename(output_path)
        if self._entity_count > 0:
            mode = self._current_mode()
            action = "anonymisiert" if mode == MODE_ANONYMIZE else "pseudonymisiert"
            detail = f"{self._entity_count} Entitäten {action}  →  {out_name}"
        else:
            detail = f"Keine PII-Daten gefunden  →  Kopie erstellt als {out_name}"

        self.drop_zone.set_state(DropZone.STATE_SUCCESS, detail)
        self.file_label.setText(out_name)
        self.open_folder_btn.setVisible(True)
        self.statusBar().showMessage(f"Gespeichert: {output_path}")

        # Open the PDF in the default viewer
        self._open_pdf(output_path)

        # Reset to idle after 8 seconds so the user can drop the next file
        QTimer.singleShot(8000, self._reset_to_idle)

    def _open_pdf(self, path: str):
        """Open the PDF in the system's default PDF viewer."""
        try:
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", path])
            else:
                import subprocess
                subprocess.Popen(["xdg-open", path])
        except Exception:
            pass  # silently ignore if no viewer is available

    def on_error(self, msg: str):
        self._set_processing(False)
        self.drop_zone.set_state(DropZone.STATE_ERROR)
        self.statusBar().showMessage("Fehler bei der Verarbeitung")

        QMessageBox.critical(
            self,
            "Fehler",
            f"Bei der Verarbeitung ist ein Fehler aufgetreten:\n\n{msg}",
        )

        # Reset to idle after closing the error dialog
        QTimer.singleShot(500, self._reset_to_idle)

    def _reset_to_idle(self):
        if self.worker and self.worker.isRunning():
            return  # still processing a new file
        if self.drop_zone._state in (DropZone.STATE_SUCCESS, DropZone.STATE_ERROR):
            self.drop_zone.set_state(DropZone.STATE_IDLE)
            self.file_label.setText("Keine Datei ausgewählt")
            self.current_pdf = None
            self._update_statusbar_idle()


# ---------------------------------------------------------------------------
# Dependency check & entry point
# ---------------------------------------------------------------------------

def _check_dependencies() -> str | None:
    missing = []
    optional_missing = []
    try:
        import fitz
    except ImportError:
        missing.append("PyMuPDF  (pip install PyMuPDF)")
    try:
        import openai
    except ImportError:
        missing.append("openai  (pip install openai)")
    if _import_error is not None and not missing:
        missing.append(str(_import_error))
    # Optional dependencies (warn but don't block)
    try:
        import ocrmypdf
    except ImportError:
        optional_missing.append("ocrmypdf  (pip install ocrmypdf – für OCR-Unterstützung)")
    try:
        import docx
    except ImportError:
        optional_missing.append("python-docx  (pip install python-docx – für Word-Dateien)")
    if missing:
        msg = "Fehlende Abhängigkeiten:\n\n" + "\n".join(f"  •  {m}" for m in missing)
        if optional_missing:
            msg += "\n\nOptional (für erweiterte Formate):\n" + "\n".join(
                f"  •  {m}" for m in optional_missing
            )
        return msg
    return None


def run_app():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Check dependencies early
    dep_err = _check_dependencies()
    if dep_err:
        QMessageBox.critical(
            None,
            "Fehlende Pakete",
            f"{dep_err}\n\nBitte installieren Sie die fehlenden Pakete:\n"
            f"  pip install -r requirements.txt",
        )
        sys.exit(1)

    # Dark palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(BG_DARK))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Base, QColor(BG_CARD))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(BG_SURFACE))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Button, QColor(BG_SURFACE))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(BLUE))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(BG_DARK))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(BG_CARD))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(TEXT_MUTED))
    app.setPalette(palette)

    # Default font
    font = app.font()
    font.setFamily("Arial, Segoe UI, SF Pro Display, system-ui, sans-serif")
    font.setPointSize(10)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
