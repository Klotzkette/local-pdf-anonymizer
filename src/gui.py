"""
PDF Anonymizer – Modern PyQt6 GUI with drag & drop, local AI model management,
and one-click anonymisation workflow.

Design: Soft blue-teal tones with black accents, Arial font, no bold text.
Sober and refined – Swiss-style minimalism.
"""

import os
import subprocess
import sys
import json
import traceback
from pathlib import Path
from typing import Optional

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
)
from PyQt6.QtCore import (
    Qt,
    QThread,
    pyqtSignal,
    QSize,
    QSettings,
    QTimer,
    QPropertyAnimation,
    QEasingCurve,
    pyqtProperty,
)
from PyQt6.QtGui import (
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QPalette,
    QColor,
    QMouseEvent,
)
from PyQt6.QtWidgets import QGraphicsDropShadowEffect, QGraphicsOpacityEffect

import logging
import time as _time

logger = logging.getLogger("pdf_anonymizer")

try:
    from ai_engine import (
        detect_entities, assign_variables, generate_natural_replacements,
        detect_entities_regex, release_model,
        MODE_ANONYMIZE, MODE_PSEUDO_VARS, MODE_PSEUDO_NATURAL,
        INTENSITY_HARD,
        SCOPE_NAMES_ONLY, SCOPE_ALL,
        is_model_downloaded, download_model,
        is_model_loaded, preload_model, wait_for_preload,
        load_model_with_progress, ModelEngine,
        MODEL_DISPLAY_NAME,
    )
    from pdf_processor import (
        extract_text, redact_pdf, get_page_count,
        prepare_input, validate_input_file, SUPPORTED_EXTENSIONS,
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
    MODEL_DISPLAY_NAME = "Qwen3.5-9B"
    def is_model_downloaded(): return False
    def download_model(progress_callback=None): pass
    def is_model_loaded(): return False
    def preload_model(): pass
    def wait_for_preload(timeout=120.0): return False
    def load_model_with_progress(progress_cb=None): return False
    class ModelEngine:
        def __init__(self, progress_cb=None): pass
        def load(self): return False
else:
    _import_error = None


# ---------------------------------------------------------------------------
# Settings persistence
# ---------------------------------------------------------------------------

SETTINGS_ORG = "toms_super_simple_pdf_anonymizer"
SETTINGS_APP = "toms_super_simple_pdf_anonymizer"


def _settings() -> QSettings:
    return QSettings(SETTINGS_ORG, SETTINGS_APP)


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
# Colour palette – refined blue tones, clear status colours
# ---------------------------------------------------------------------------

BG_DARK         = "#EEF2F9"         # soft lavender-grey background
BG_CARD         = "#F7F9FC"         # almost-white card surface
BG_SURFACE      = "#E2E9F3"         # subtle blue for inputs/bars
BG_HOVER        = "#D4DEEE"         # hover highlight

ACCENT          = "#2563EB"         # vivid blue primary
ACCENT_HOVER    = "#1D4ED8"         # deeper blue on hover
ACCENT_SOFT     = "#6B9AE8"         # lighter muted blue for secondary
ACCENT_GLOW     = "#93B5F5"         # very light blue for glows/shadows

BORDER          = "#C8D6EA"         # softer blue-grey border
BORDER_FOCUS    = "#3B7DD8"         # strong blue on focus

TEXT_PRIMARY    = "#0F172A"         # near-black with blue tint
TEXT_SECONDARY  = "#475569"         # slate secondary
TEXT_MUTED      = "#94A3B8"         # slate muted

SUCCESS         = "#16A34A"         # clear green for success
SUCCESS_BG      = "#F0FDF4"         # very light green tint
SUCCESS_BORDER  = "#86EFAC"         # soft green border

ERROR           = "#DC2626"         # red for errors
ERROR_BG        = "#FEF2F2"         # very light red tint
ERROR_BORDER    = "#FCA5A5"         # soft red border


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------

STYLESHEET = f"""
/* ── Global ────────────────────────────────────────────── */
* {{
    font-family: "SF Pro Display", "Segoe UI", "Helvetica Neue", "Arial", sans-serif;
    font-weight: normal;
}}
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
    font-size: 22px;
    letter-spacing: -0.3px;
}}
QLabel#titleAccent {{
    color: {ACCENT};
    font-size: 22px;
    letter-spacing: -0.3px;
}}
QLabel#subtitleLabel {{
    color: {TEXT_MUTED};
    font-size: 12px;
    line-height: 1.6;
}}
QLabel#dropIcon {{
    font-size: 48px;
    background: transparent;
}}
QLabel#dropLabel {{
    color: {TEXT_PRIMARY};
    font-size: 15px;
}}
QLabel#dropHint {{
    color: {TEXT_MUTED};
    font-size: 12px;
}}
QLabel#fileLabel {{
    color: {TEXT_SECONDARY};
    font-size: 12px;
    padding: 4px 0px;
}}
QLabel#statusLabel {{
    color: {TEXT_SECONDARY};
    font-size: 12px;
}}
QLabel#stepLabel {{
    color: {ACCENT};
    font-size: 12px;
}}
QLabel#providerPill {{
    color: {ACCENT};
    background-color: {BG_CARD};
    border: 1px solid {ACCENT_GLOW};
    border-radius: 14px;
    padding: 4px 12px;
    font-size: 11px;
}}
QLabel#successLabel {{
    color: {SUCCESS};
    font-size: 13px;
}}
QLabel#errorLabel {{
    color: {ERROR};
    font-size: 13px;
}}
QLabel#versionLabel {{
    color: {TEXT_MUTED};
    font-size: 10px;
    background: transparent;
}}

/* ── Buttons ───────────────────────────────────────────── */
QPushButton {{
    background-color: {ACCENT};
    color: #FFFFFF;
    border: none;
    border-radius: 16px;
    padding: 10px 28px;
    font-size: 13px;
}}
QPushButton:hover {{
    background-color: {ACCENT_HOVER};
}}
QPushButton:pressed {{
    background-color: {ACCENT_SOFT};
}}
QPushButton:disabled {{
    background-color: {BG_SURFACE};
    color: {TEXT_MUTED};
    border-radius: 16px;
}}
QPushButton#settingsBtn {{
    background-color: transparent;
    color: {TEXT_MUTED};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 6px 16px;
    font-size: 11px;
}}
QPushButton#settingsBtn:hover {{
    color: {ACCENT};
    background-color: {BG_CARD};
    border-color: {ACCENT_GLOW};
}}
QPushButton#selectBtn {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 9px 22px;
    font-size: 13px;
}}
QPushButton#selectBtn:hover {{
    background-color: {BG_HOVER};
    border-color: {ACCENT_SOFT};
}}
QPushButton#selectBtn:disabled {{
    color: {TEXT_MUTED};
    border-color: {BORDER};
    background-color: {BG_SURFACE};
}}
QPushButton#openFolderBtn {{
    background-color: {SUCCESS_BG};
    color: {SUCCESS};
    border: 1px solid {SUCCESS_BORDER};
    border-radius: 14px;
    padding: 6px 16px;
    font-size: 12px;
}}
QPushButton#openFolderBtn:hover {{
    background-color: #DCFCE7;
    border-color: {SUCCESS};
}}

/* ── Progress ──────────────────────────────────────────── */
QProgressBar {{
    border: none;
    border-radius: 10px;
    text-align: center;
    color: #FFFFFF;
    background-color: {BG_SURFACE};
    min-height: 20px;
    max-height: 20px;
    font-size: 10px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {ACCENT}, stop:0.5 {ACCENT_SOFT}, stop:1 {ACCENT});
    border-radius: 10px;
}}

/* ── Inputs ────────────────────────────────────────────── */
QComboBox {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 10px 14px;
    font-size: 13px;
}}
QComboBox:focus {{
    border-color: {BORDER_FOCUS};
}}
QComboBox QAbstractItemView {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    selection-background-color: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    outline: none;
    padding: 4px;
}}
QComboBox::drop-down {{
    border: none;
    padding-right: 10px;
}}
QLineEdit {{
    background-color: #FFFFFF;
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 10px 14px;
    font-size: 13px;
    selection-background-color: {BG_HOVER};
}}
QLineEdit:focus {{
    border-color: {ACCENT};
}}
QLineEdit[valid="true"] {{
    border-color: {SUCCESS};
}}

/* ── Groups ────────────────────────────────────────────── */
QGroupBox {{
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 16px;
    margin-top: 14px;
    padding: 22px 18px 14px 18px;
    font-size: 13px;
    background-color: {BG_CARD};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 18px;
    padding: 0 10px;
    color: {TEXT_SECONDARY};
    font-size: 12px;
}}

/* ── Dialog ────────────────────────────────────────────── */
QDialog {{
    background-color: {BG_DARK};
}}

/* ── Status bar ────────────────────────────────────────── */
QStatusBar {{
    background-color: {BG_CARD};
    color: {TEXT_MUTED};
    font-size: 10px;
    border-top: 1px solid {BORDER};
    padding: 3px 12px;
}}

/* ── Drop zone ─────────────────────────────────────────── */
QFrame#dropZone {{
    background-color: {BG_CARD};
    border: 2px dashed {BORDER};
    border-radius: 24px;
}}
QFrame#dropZone:hover {{
    border-color: {ACCENT_SOFT};
    background-color: #FFFFFF;
}}
QFrame#dropZone[dragOver="true"] {{
    border-color: {ACCENT};
    border-style: solid;
    border-width: 2px;
    background-color: #EBF4FF;
}}
QFrame#dropZone[processing="true"] {{
    border-color: {ACCENT_GLOW};
    border-style: solid;
    background-color: {BG_CARD};
}}
QFrame#dropZone[success="true"] {{
    border-color: {SUCCESS_BORDER};
    border-style: solid;
    background-color: {SUCCESS_BG};
}}
QFrame#dropZone[error="true"] {{
    border-color: {ERROR_BORDER};
    border-style: solid;
    background-color: {ERROR_BG};
}}

/* ── Scrollbar (macOS-style) ──────────────────────────── */
QScrollBar:vertical {{
    background-color: transparent;
    width: 6px;
    margin: 4px 2px;
}}
QScrollBar::handle:vertical {{
    background-color: {BG_HOVER};
    border-radius: 3px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background-color: {ACCENT_SOFT};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

/* ── Tooltip ──────────────────────────────────────────── */
QToolTip {{
    background-color: {TEXT_PRIMARY};
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    padding: 6px 10px;
    font-size: 11px;
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
        self.setMinimumHeight(220)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._state = self.STATE_IDLE

        # Drop shadow for depth
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(32)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 22))
        self.setGraphicsEffect(shadow)
        self._shadow = shadow

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(8)

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
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setMinimumWidth(320)
        self.progress_bar.setMaximumWidth(400)
        self.progress_bar.setFormat("%p %")
        layout.addWidget(self.progress_bar, alignment=Qt.AlignmentFlag.AlignCenter)

        # Idle pulse animation for the shadow
        self._pulse_anim = QPropertyAnimation(shadow, b"blurRadius")
        self._pulse_anim.setDuration(2400)
        self._pulse_anim.setStartValue(24)
        self._pulse_anim.setEndValue(40)
        self._pulse_anim.setEasingCurve(QEasingCurve.Type.InOutSine)
        self._pulse_anim.setLoopCount(-1)  # infinite

        self.set_state(self.STATE_IDLE)

    def set_state(self, state: str, detail: str = ""):
        self._state = state
        # Reset all dynamic properties
        for prop in ("dragOver", "processing", "success", "error"):
            self.setProperty(prop, False)

        # Stop pulse animation by default
        self._pulse_anim.stop()

        if state == self.STATE_IDLE:
            self.icon_label.setText("\u2B06")  # upward arrow
            self.primary_label.setText("Datei hier ablegen")
            self.secondary_label.setText("PDF, Word, JPG \u2013 oder klicken zum Ausw\u00e4hlen")
            self.secondary_label.setVisible(True)
            self.step_label.setVisible(False)
            self.progress_bar.setVisible(False)
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setAcceptDrops(True)
            # Subtle shadow pulse on idle
            self._shadow.setColor(QColor(37, 99, 235, 18))  # ACCENT with low alpha
            self._pulse_anim.start()

        elif state == self.STATE_PROCESSING:
            self.setProperty("processing", True)
            self._shadow.setColor(QColor(37, 99, 235, 30))
            self._shadow.setBlurRadius(28)
            self.icon_label.setText("\u2699")  # gear
            self.primary_label.setText("Wird verarbeitet \u2026")
            self.secondary_label.setVisible(False)
            self.step_label.setVisible(True)
            self.step_label.setText(detail or "Initialisiere \u2026")
            self.progress_bar.setVisible(True)
            self.setCursor(Qt.CursorShape.WaitCursor)
            self.setAcceptDrops(False)

        elif state == self.STATE_SUCCESS:
            self.setProperty("success", True)
            self._shadow.setColor(QColor(22, 163, 74, 30))  # green shadow
            self._shadow.setBlurRadius(32)
            self.icon_label.setText("\u2713")  # check mark (simple)
            self.primary_label.setText("Anonymisierung abgeschlossen")
            self.primary_label.setStyleSheet(f"color: {SUCCESS}; font-size: 15px;")
            self.secondary_label.setText(detail or "")
            self.secondary_label.setVisible(bool(detail))
            self.step_label.setVisible(False)
            self.progress_bar.setVisible(False)
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setAcceptDrops(True)

        elif state == self.STATE_ERROR:
            self.setProperty("error", True)
            self._shadow.setColor(QColor(220, 38, 38, 25))  # red shadow
            self._shadow.setBlurRadius(28)
            self.icon_label.setText("\u2717")  # ballot X
            self.primary_label.setText("Fehler aufgetreten")
            self.primary_label.setStyleSheet(f"color: {ERROR}; font-size: 15px;")
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
            self.primary_label.setStyleSheet(f"color: {SUCCESS}; font-size: 15px;")
        elif state == self.STATE_ERROR:
            self.primary_label.setStyleSheet(f"color: {ERROR}; font-size: 15px;")

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
# Global model-load progress state  (read by GUI via QTimer polling)
# ---------------------------------------------------------------------------

_current_progress = {"phase": "", "value": 0.0, "running": False, "error": ""}


class ModelLoadWorker(QThread):
    """Loads the AI model using the 3-phase pipeline with progress signals."""

    phase_progress = pyqtSignal(str, float)   # (phase, value 0.0-1.0)
    finished_ok = pyqtSignal()
    finished_err = pyqtSignal(str)

    _PHASE_LABELS = {
        "io":   "Readahead …",
        "init": "Modell laden …",
    }

    def _on_progress(self, phase: str, value: float) -> None:
        _current_progress["phase"] = phase
        _current_progress["value"] = value
        self.phase_progress.emit(phase, value)

    def run(self) -> None:
        _current_progress.update(phase="", value=0.0, running=True, error="")
        try:
            ok = load_model_with_progress(progress_cb=self._on_progress)
            if ok:
                _current_progress["running"] = False
                self.finished_ok.emit()
            else:
                _current_progress.update(running=False, error="Modell konnte nicht geladen werden")
                self.finished_err.emit("Modell konnte nicht geladen werden")
        except Exception as exc:
            logger.error("ModelLoadWorker fehlgeschlagen", exc_info=True)
            _current_progress.update(running=False, error=str(exc))
            self.finished_err.emit(str(exc))


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
        self._temp_pdf: Optional[str] = None

    def run(self):
        t_start = _time.monotonic()
        use_regex_fallback = False

        try:
            # Step 0 – ensure model is in RAM (instant if preloaded at startup)
            self.step.emit("Modell laden …")
            self.progress.emit(1)

            try:
                from ai_engine import is_model_loaded, wait_for_preload
                if is_model_loaded():
                    self.status.emit(f"KI-Modell ({MODEL_DISPLAY_NAME}) bereit")
                else:
                    # Wait for any ongoing background preload first
                    wait_for_preload(timeout=120.0)

                    if not is_model_loaded():
                        # Load with phase-aware progress reporting
                        _phase_labels = ModelLoadWorker._PHASE_LABELS

                        def _on_phase(phase: str, value: float) -> None:
                            label = _phase_labels.get(phase, phase)
                            pct = int(value * 100)
                            self.status.emit(f"{label} ({pct} %)")

                        ok = load_model_with_progress(progress_cb=_on_phase)
                        if not ok:
                            logger.warning("Modell konnte nicht geladen werden, "
                                           "verwende Regex-Fallback")
                            self.status.emit("KI-Modell nicht verfügbar – verwende einfache Erkennung …")
                            use_regex_fallback = True
            except Exception:
                logger.error("Modell-Laden fehlgeschlagen", exc_info=True)
                self.status.emit("KI-Modell nicht verfügbar – verwende einfache Erkennung …")
                use_regex_fallback = True

            # Step 1 – prepare input (convert / OCR if needed)
            self.step.emit("Schritt 1/5  –  Datei vorbereiten")
            self.status.emit("Eingabedatei wird vorbereitet …")
            self.progress.emit(2)

            try:
                pdf_path = prepare_input(
                    self.input_path,
                    api_key=self.api_key,
                    status_callback=lambda msg: self.status.emit(msg),
                )
            except Exception as e:
                self.finished_err.emit(
                    f"Datei-Fehler: {e}\n\n"
                    "Die Eingabedatei konnte nicht verarbeitet werden.\n"
                    "Bitte prüfen Sie, ob die Datei gültig und nicht beschädigt ist."
                )
                return

            if pdf_path != self.input_path:
                self._temp_pdf = pdf_path  # remember for cleanup

            # Step 2 – extract text
            self.step.emit("Schritt 2/5  –  Text extrahieren")
            self.status.emit("Text wird aus dem PDF extrahiert …")
            self.progress.emit(8)

            try:
                text = extract_text(pdf_path)
            except Exception as e:
                self._cleanup_temp()
                self.finished_err.emit(
                    f"Text-Extraktion fehlgeschlagen: {e}\n\n"
                    "Der Text konnte nicht aus dem PDF gelesen werden.\n"
                    "Möglicherweise ist die Datei beschädigt oder enthält keinen Text."
                )
                return

            # Step 3 – AI entity detection (with regex fallback)
            provider = "regex" if use_regex_fallback else self.provider
            if use_regex_fallback:
                self.step.emit("Schritt 3/5  –  Muster-Erkennung")
                self.status.emit("Einfache Muster-Erkennung läuft …")
            else:
                self.step.emit("Schritt 3/5  –  KI-Analyse")
                self.status.emit("KI analysiert den Text …")
            self.progress.emit(12)

            def _ai_progress(pct):
                self.progress.emit(12 + int(pct * 0.28))

            try:
                entities = detect_entities(
                    provider, self.api_key, text,
                    progress_callback=_ai_progress,
                    intensity=INTENSITY_HARD,
                    scope=self.scope,
                )
            except Exception:
                logger.error("Entitäten-Erkennung fehlgeschlagen, "
                             "Fallback auf Regex", exc_info=True)
                self.status.emit("KI-Fehler – verwende einfache Erkennung …")
                entities = detect_entities_regex(text, self.scope)

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
            if self.mode == MODE_PSEUDO_NATURAL and not use_regex_fallback:
                self.step.emit("Schritt 4/5  –  Natürliche Ersetzungen generieren")
                self.status.emit(f"{len(entities)} Entitäten erkannt – generiere Ersetzungen …")
                self.progress.emit(42)
                try:
                    replacements = generate_natural_replacements(
                        self.provider, self.api_key, entities,
                    )
                except Exception:
                    logger.error("Ersetzungs-Generierung fehlgeschlagen", exc_info=True)
                    self.status.emit("Ersetzungs-Generierung fehlgeschlagen, "
                                     "verwende Variablen-Modus …")
                    replacements = None
            else:
                self.step.emit("Schritt 4/5  –  Variablen zuweisen")
                self.status.emit(f"{len(entities)} Entitäten erkannt …")
                self.progress.emit(42)

            entity_map = assign_variables(entities, mode=self.mode, replacements=replacements)

            # Step 5 – redact PDF
            mode_label = {
                MODE_ANONYMIZE: "anonymisieren",
                MODE_PSEUDO_NATURAL: "pseudonymisieren",
            }.get(self.mode, "verarbeiten")
            self.step.emit(f"Schritt 5/5  –  PDF {mode_label}")
            self.status.emit("PDF wird geschrieben …")

            def _pdf_progress(pct):
                self.progress.emit(45 + int(pct * 0.50))

            try:
                redact_pdf(
                    pdf_path, self.output_path, entity_map,
                    mode=self.mode, progress_callback=_pdf_progress,
                    api_key=self.api_key,
                )
            except Exception as e:
                self._cleanup_temp()
                self.finished_err.emit(
                    f"PDF-Fehler: {e}\n\n"
                    "Das PDF konnte nicht geschwärzt werden.\n"
                    "Bitte prüfen Sie Speicherplatz und Schreibrechte."
                )
                return

            self._cleanup_temp()
            self.progress.emit(100)
            elapsed = _time.monotonic() - t_start
            logger.info("Verarbeitung abgeschlossen in %.1f Sekunden", elapsed)
            self.finished_ok.emit(self.output_path)

        except Exception as e:
            self._cleanup_temp()
            logger.error("Unerwarteter Fehler in AnonymizeWorker", exc_info=True)
            self.finished_err.emit(
                f"Unerwarteter Fehler: {e}\n\n"
                "Bitte prüfen Sie die Log-Datei für Details:\n"
                f"{os.path.expanduser('~/.cache/pdf_anonymizer/anonymizer.log')}"
            )

    def _cleanup_temp(self):
        """Remove temporary PDF created during conversion/OCR."""
        if self._temp_pdf:
            try:
                os.unlink(self._temp_pdf)
            except OSError:
                pass
            self._temp_pdf = None


# ---------------------------------------------------------------------------
# Model download worker thread
# ---------------------------------------------------------------------------

class ModelDownloadWorker(QThread):
    """Downloads Qwen3.5-9B GGUF from HuggingFace in the background."""

    progress = pyqtSignal(int)    # -1 = indeterminate, 0-100 = percentage
    status   = pyqtSignal(str)
    finished_ok  = pyqtSignal()
    finished_err = pyqtSignal(str)

    def run(self):
        try:
            def _cb(pct: int, msg: str):
                self.progress.emit(pct)
                self.status.emit(msg)

            download_model(progress_callback=_cb)
            self.finished_ok.emit()
        except Exception as e:
            import traceback
            self.finished_err.emit(f"{e}\n\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Settings / Model management dialog
# ---------------------------------------------------------------------------

class SettingsDialog(QDialog):
    """KI-Modell-Verwaltung: Download-Status und Download-Button."""

    model_status_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("KI-Modell")
        self.setMinimumWidth(500)
        self.setStyleSheet(STYLESHEET)
        self._download_worker: ModelDownloadWorker | None = None

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(32, 28, 32, 24)

        # -- Header --
        header = QLabel(f"\u2728  KI-Modell: {MODEL_DISPLAY_NAME}")
        header.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 18px; letter-spacing: -0.3px;"
        )
        layout.addWidget(header)

        desc = QLabel(
            "Das Modell wird lokal auf Ihrem Ger\u00e4t gespeichert und ausgef\u00fchrt.\n"
            "Es wird kein API-Key ben\u00f6tigt und keine Daten werden \u00fcbertragen."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px; line-height: 1.5;")
        layout.addWidget(desc)

        layout.addSpacing(4)

        # -- Model info box --
        info_box = QGroupBox("Modell-Informationen")
        info_layout = QFormLayout(info_box)
        info_layout.setSpacing(8)
        info_layout.addRow(
            "Modell:", QLabel(f"{MODEL_DISPLAY_NAME} (9B Parameter, Q4_K_M)")
        )
        info_layout.addRow("Kontext:", QLabel("262 000 Tokens (bis 1M erweiterbar)"))
        info_layout.addRow("Typ:", QLabel("Sprachmodell, lokal"))
        info_layout.addRow(
            "Speicherbedarf:", QLabel("\u223C 6 GB Download  \u00b7  \u223C 8 GB RAM")
        )
        layout.addWidget(info_box)

        # -- Status + progress --
        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
        layout.addWidget(self._status_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setFixedHeight(20)
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        layout.addStretch()

        # -- Buttons --
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        btn_layout.addStretch()

        self._close_btn = QPushButton("Schlie\u00dfen")
        self._close_btn.setObjectName("selectBtn")
        self._close_btn.clicked.connect(self._on_close)
        btn_layout.addWidget(self._close_btn)

        self._download_btn = QPushButton("\u2193  Modell herunterladen")
        self._download_btn.setMinimumWidth(200)
        self._download_btn.clicked.connect(self._start_download)
        btn_layout.addWidget(self._download_btn)

        layout.addLayout(btn_layout)

        self._refresh_status()

    # ------------------------------------------------------------------

    def _refresh_status(self):
        if is_model_downloaded():
            self._status_label.setText(
                f"\u2713  {MODEL_DISPLAY_NAME} ist heruntergeladen und einsatzbereit."
            )
            self._status_label.setStyleSheet(
                f"color: {SUCCESS}; font-size: 13px;"
            )
            self._download_btn.setVisible(False)
        else:
            self._status_label.setText(
                f"Das Modell wurde noch nicht heruntergeladen.\n"
                f"Dr\u00fccken Sie \u201eModell herunterladen\u201c, um zu starten."
            )
            self._status_label.setStyleSheet(
                f"color: {TEXT_SECONDARY}; font-size: 12px;"
            )
            self._download_btn.setVisible(True)

    def _start_download(self):
        if self._download_worker and self._download_worker.isRunning():
            return

        self._download_btn.setEnabled(False)
        self._close_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._status_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 12px;"
        )
        self._status_label.setText(f"Starte Download von {MODEL_DISPLAY_NAME} …")

        self._download_worker = ModelDownloadWorker()
        self._download_worker.progress.connect(self._on_dl_progress)
        self._download_worker.status.connect(self._on_dl_status)
        self._download_worker.finished_ok.connect(self._on_dl_ok)
        self._download_worker.finished_err.connect(self._on_dl_err)
        self._download_worker.finished.connect(self._download_worker.deleteLater)
        self._download_worker.start()

    def _on_dl_progress(self, pct: int):
        if pct < 0:
            # Indeterminate
            self._progress_bar.setRange(0, 0)
        else:
            self._progress_bar.setRange(0, 100)
            self._progress_bar.setValue(pct)

    def _on_dl_status(self, msg: str):
        self._status_label.setText(msg)

    def _on_dl_ok(self):
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(100)
        self._close_btn.setEnabled(True)
        self._refresh_status()
        # Immediately start preloading the model into RAM
        preload_model()
        self.model_status_changed.emit()

    def _on_dl_err(self, msg: str):
        self._progress_bar.setVisible(False)
        self._download_btn.setEnabled(True)
        self._close_btn.setEnabled(True)
        self._status_label.setText(f"Fehler beim Download:\n{msg.splitlines()[0]}")
        self._status_label.setStyleSheet(f"color: {ERROR}; font-size: 12px;")

    def _on_close(self):
        if self._download_worker and self._download_worker.isRunning():
            # Ask for confirmation before interrupting download
            reply = QMessageBox.question(
                self,
                "Download l\u00e4uft",
                "Der Download l\u00e4uft noch. Wirklich abbrechen?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            self._download_worker.terminate()
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
            chip.setFixedHeight(36)
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
                    f"QPushButton {{ background-color: {ACCENT}; color: #FFFFFF; "
                    f"border: none; border-radius: 14px; padding: 6px 18px; "
                    f"font-size: 12px; }}"
                )
            else:
                chip.setStyleSheet(
                    f"QPushButton {{ background-color: #FFFFFF; color: {TEXT_SECONDARY}; "
                    f"border: 1px solid {BORDER}; border-radius: 14px; padding: 6px 18px; "
                    f"font-size: 12px; }}"
                    f"QPushButton:hover {{ background-color: #EBF4FF; "
                    f"border-color: {ACCENT_SOFT}; color: {ACCENT}; }}"
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
        self.selected_mode: Optional[str] = None
        self.selected_scope: str = load_scope()

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(32, 28, 32, 24)

        # Header
        header = QLabel("Verarbeitungsoptionen")
        header.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 17px; letter-spacing: -0.2px;"
        )
        layout.addWidget(header)
        layout.addSpacing(4)

        # -- Scope section --
        scope_label = QLabel("Umfang")
        scope_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px; "
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
        divider.setStyleSheet(f"background-color: {BORDER}; border-radius: 1px;")
        layout.addWidget(divider)

        layout.addSpacing(4)

        # -- Mode section --
        mode_label = QLabel("Modus  (zum Starten anklicken)")
        mode_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px; "
            f"text-transform: uppercase; letter-spacing: 1px;"
        )
        layout.addWidget(mode_label)
        layout.addSpacing(2)

        saved_mode = load_mode()

        # Mode icons
        mode_icons = {
            MODE_ANONYMIZE: "\u2588\u2588",  # solid blocks = redaction
            MODE_PSEUDO_NATURAL: "\u21C4",    # arrows = replacement
        }

        for mode_key, title, desc in _MODE_OPTIONS:
            is_saved = mode_key == saved_mode
            card = QFrame()
            card.setCursor(Qt.CursorShape.PointingHandCursor)
            border_col = ACCENT if is_saved else BORDER
            bg_col = "#EBF4FF" if is_saved else BG_CARD
            card.setStyleSheet(f"""
                QFrame {{
                    background-color: {bg_col};
                    border: {"2" if is_saved else "1"}px solid {border_col};
                    border-radius: 14px;
                    padding: 14px 18px;
                }}
                QFrame:hover {{
                    border-color: {ACCENT};
                    background-color: #EBF4FF;
                }}
            """)

            # Add shadow to card
            card_shadow = QGraphicsDropShadowEffect(card)
            card_shadow.setBlurRadius(16)
            card_shadow.setOffset(0, 2)
            card_shadow.setColor(QColor(0, 0, 0, 15))
            card.setGraphicsEffect(card_shadow)

            card_layout = QHBoxLayout(card)
            card_layout.setSpacing(14)
            card_layout.setContentsMargins(0, 0, 0, 0)

            # Icon on the left
            icon = QLabel(mode_icons.get(mode_key, ""))
            icon.setStyleSheet(
                f"color: {ACCENT}; font-size: 18px; "
                f"border: none; background: transparent;"
            )
            icon.setFixedWidth(30)
            icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card_layout.addWidget(icon)

            # Text on the right
            text_layout = QVBoxLayout()
            text_layout.setSpacing(2)

            title_label = QLabel(title)
            title_label.setStyleSheet(
                f"color: {TEXT_PRIMARY}; font-size: 13px; "
                f"border: none; background: transparent;"
            )
            text_layout.addWidget(title_label)

            desc_label = QLabel(desc)
            desc_label.setStyleSheet(
                f"color: {TEXT_MUTED}; font-size: 11px; "
                f"border: none; background: transparent;"
            )
            desc_label.setWordWrap(True)
            text_layout.addWidget(desc_label)

            card_layout.addLayout(text_layout, stretch=1)

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
        self.setMinimumSize(620, 520)
        self.resize(700, 600)
        self.setStyleSheet(STYLESHEET)

        self.worker = None
        self.current_pdf: Optional[str] = None
        self._last_output: Optional[str] = None
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
        t1 = QLabel("Tom\u2019s Super Simple ")
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

        header.addSpacing(6)

        settings_btn = QPushButton("\u2699  Einstellungen")
        settings_btn.setObjectName("settingsBtn")
        settings_btn.clicked.connect(self.open_settings)
        header.addWidget(settings_btn, alignment=Qt.AlignmentFlag.AlignVCenter)

        main_layout.addLayout(header)
        main_layout.addSpacing(2)

        # Subtitle
        subtitle = QLabel(
            "KI-gest\u00fctzte Anonymisierung personenbezogener Daten  \u00b7  "
            "PDF, Word, JPG  \u00b7  OCR bei Bedarf"
        )
        subtitle.setObjectName("subtitleLabel")
        subtitle.setWordWrap(True)
        main_layout.addWidget(subtitle)
        main_layout.addSpacing(14)

        # ── Drop zone ──
        self.drop_zone = DropZone()
        self.drop_zone.file_dropped.connect(self.on_file_selected)
        self.drop_zone.clicked.connect(self.browse_pdf)
        main_layout.addWidget(self.drop_zone, stretch=1)
        main_layout.addSpacing(12)

        # ── Bottom bar (file info + buttons) ──
        bottom = QHBoxLayout()
        bottom.setSpacing(10)

        self.file_label = QLabel("")
        self.file_label.setObjectName("fileLabel")
        self.file_label.setWordWrap(True)
        bottom.addWidget(self.file_label, stretch=1)

        self.open_folder_btn = QPushButton("\u2197  Ordner \u00f6ffnen")
        self.open_folder_btn.setObjectName("openFolderBtn")
        self.open_folder_btn.setVisible(False)
        self.open_folder_btn.clicked.connect(self._open_output_folder)
        bottom.addWidget(self.open_folder_btn)

        self.select_btn = QPushButton("Datei ausw\u00e4hlen")
        self.select_btn.setObjectName("selectBtn")
        self.select_btn.clicked.connect(self.browse_pdf)
        bottom.addWidget(self.select_btn)

        main_layout.addLayout(bottom)
        main_layout.addSpacing(2)

        # ── Status bar ──
        self.statusBar().showMessage("Bereit  \u00b7  v3.2")
        self._update_statusbar_idle()

        # ── Background model preload ──
        # If model is downloaded, start loading it into RAM immediately
        # so it's ready when the user drops a file.
        if is_model_downloaded() and not is_model_loaded():
            preload_model()
            # Poll until model is loaded, then update the UI
            self._preload_timer = QTimer(self)
            self._preload_timer.setInterval(1000)
            self._preload_timer.timeout.connect(self._check_preload)
            self._preload_timer.start()

    def _check_preload(self):
        """Called by timer to update UI once background preload finishes."""
        if is_model_loaded():
            self._preload_timer.stop()
            self._update_provider_pill()
            self._update_statusbar_idle()

    # -- Helpers --

    def _update_provider_pill(self):
        if is_model_loaded():
            self.provider_pill.setText(f"{MODEL_DISPLAY_NAME}  \u00b7  Bereit")
            self.provider_pill.setStyleSheet(
                f"color: {SUCCESS}; background-color: {SUCCESS_BG}; "
                f"border: 1px solid {SUCCESS_BORDER}; "
                f"border-radius: 14px; padding: 4px 12px; font-size: 11px;"
            )
        elif is_model_downloaded():
            self.provider_pill.setText(f"{MODEL_DISPLAY_NAME}  \u00b7  Wird geladen \u2026")
            self.provider_pill.setStyleSheet("")  # default stylesheet
        else:
            self.provider_pill.setText(f"{MODEL_DISPLAY_NAME}  \u00b7  nicht geladen")
            self.provider_pill.setStyleSheet(
                f"color: {ERROR}; background-color: {ERROR_BG}; "
                f"border: 1px solid {ERROR_BORDER}; "
                f"border-radius: 14px; padding: 4px 12px; font-size: 11px;"
            )

    def _update_statusbar_idle(self):
        if is_model_loaded():
            self.statusBar().showMessage(
                f"Bereit  \u00b7  {MODEL_DISPLAY_NAME} im RAM  \u00b7  PDF ablegen oder ausw\u00e4hlen  \u00b7  v3.2"
            )
        elif is_model_downloaded():
            self.statusBar().showMessage(
                f"Modell wird geladen \u2026  \u00b7  {MODEL_DISPLAY_NAME}  \u00b7  v3.2"
            )
        else:
            self.statusBar().showMessage(
                f"KI-Modell nicht geladen  \u00b7  Bitte unter \u2699 Einstellungen herunterladen  \u00b7  v3.2"
            )

    def _current_mode(self) -> str:
        return self._selected_mode

    def _set_processing(self, active: bool):
        """Lock/unlock UI during processing."""
        self.select_btn.setEnabled(not active)

    def _open_output_folder(self):
        if self._last_output:
            folder = os.path.dirname(self._last_output)
            self._open_path(folder)

    @staticmethod
    def _open_path(path: str):
        """Open a file or folder in the system's default application."""
        try:
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception:
            pass

    # -- Slots --

    def open_settings(self):
        dlg = SettingsDialog(self)
        dlg.model_status_changed.connect(self._update_provider_pill)
        dlg.model_status_changed.connect(self._update_statusbar_idle)
        dlg.exec()
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

        # Check that the model has been downloaded
        if not is_model_downloaded():
            reply = QMessageBox.question(
                self,
                "KI-Modell nicht geladen",
                f"Das KI-Modell ({MODEL_DISPLAY_NAME}) wurde noch nicht\n"
                f"heruntergeladen.\n\nJetzt herunterladen?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.open_settings()
            if not is_model_downloaded():
                return

        provider = "qwen"
        api_key = ""

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
        self.worker.finished.connect(self.worker.deleteLater)
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
        self._open_path(path)

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

def _check_dependencies() -> Optional[str]:
    import importlib.util

    missing = []
    optional_missing = []

    # fitz (PyMuPDF) is lightweight – safe to import eagerly
    try:
        import fitz  # noqa: F401
    except ImportError:
        missing.append("PyMuPDF  (pip install PyMuPDF)")

    # llama-cpp-python and huggingface_hub: only check whether they are
    # *findable* without actually importing them to avoid slow DLL loading.
    for pkg, install_hint in [
        ("llama_cpp", "llama-cpp-python  (pip install llama-cpp-python)"),
        ("huggingface_hub", "huggingface_hub  (pip install huggingface_hub)"),
    ]:
        if importlib.util.find_spec(pkg) is None:
            missing.append(install_hint)

    if _import_error is not None and not missing:
        missing.append(str(_import_error))

    # Optional dependencies (warn but don't block)
    if importlib.util.find_spec("ocrmypdf") is None:
        optional_missing.append("ocrmypdf  (pip install ocrmypdf – für OCR-Unterstützung)")
    if importlib.util.find_spec("docx") is None:
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

    # Application palette (blue-teal tones with black accents)
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(BG_DARK))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Base, QColor(BG_CARD))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(BG_SURFACE))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Button, QColor(BG_SURFACE))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(BG_CARD))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(TEXT_MUTED))
    app.setPalette(palette)

    # Default font – Arial, no bold
    font = QFont("Arial")
    font.setPointSize(10)
    font.setWeight(QFont.Weight.Normal)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
