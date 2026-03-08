"""
PDF Anonymizer – Entry point.

Starts the PyQt6 GUI application for AI-powered PDF anonymisation.
Installs a global exception handler so the app never crashes silently.
"""

import sys
import os

# PyInstaller with console=False sets sys.stdout/stderr to None.
# Libraries like huggingface_hub/tqdm call sys.stderr.write() internally,
# which crashes with "'NoneType' object has no attribute 'write'".
# Redirect to devnull to prevent this.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

# Ensure the src directory is on the path (needed for PyInstaller bundles).
# When frozen (--onefile / --onedir) the modules live inside _MEIPASS/src;
# when running from source they sit next to this file.
if getattr(sys, "frozen", False):
    base_dir = sys._MEIPASS
    src_dir = os.path.join(base_dir, "src")
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = base_dir

for p in (base_dir, src_dir):
    if p not in sys.path:
        sys.path.insert(0, p)


def _install_global_exception_handler():
    """Install a last-resort exception handler that logs and shows a dialog."""
    import logging

    def _handler(exc_type, exc_value, exc_tb):
        if exc_type is KeyboardInterrupt:
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return

        logger = logging.getLogger("pdf_anonymizer")
        logger.critical("Unbehandelter Fehler", exc_info=(exc_type, exc_value, exc_tb))

        # Try to show an error dialog (may fail if Qt is not running)
        try:
            from PyQt6.QtWidgets import QApplication, QMessageBox
            app = QApplication.instance()
            if app is not None:
                import traceback
                tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Icon.Critical)
                msg.setWindowTitle("Schwerwiegender Fehler")
                msg.setText(
                    "Ein unerwarteter Fehler ist aufgetreten.\n\n"
                    "Das Programm muss möglicherweise neu gestartet werden."
                )
                msg.setDetailedText(tb_text)
                msg.exec()
        except Exception:
            pass  # can't show dialog – at least we logged it

    sys.excepthook = _handler


_install_global_exception_handler()

from gui import run_app

if __name__ == "__main__":
    run_app()
