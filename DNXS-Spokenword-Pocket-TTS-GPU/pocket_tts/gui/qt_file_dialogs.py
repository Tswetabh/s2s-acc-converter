"""QFileDialog helpers that hide hidden files and folders by default.

Native platform dialogs often show dotfiles; we use Qt's built-in dialog and
a QDir filter without QDir.Hidden so .git / .cache / etc. stay hidden unless
the user enables "Show Hidden Files" in that dialog (when available).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from qtpy.QtCore import QDir, QTimer
from qtpy.QtWidgets import QApplication, QFileDialog, QListView, QSplitter, QSizePolicy, QWidget

# Visible entries only — do not include QDir.Hidden
_VISIBLE_ENTRIES = (
    QDir.Dirs | QDir.Files | QDir.Drives | QDir.NoDotAndDotDot
)


def _all_files_first(filter_spec: str) -> str:
    """Return filter choices with ``All Files (*)`` as the initial choice."""
    choices = [choice.strip() for choice in filter_spec.split(";;") if choice.strip()]
    all_files = "All Files (*)"
    choices = [choice for choice in choices if choice != all_files]
    return ";;".join([all_files, *choices])


def _prepare_dialog(dlg: QFileDialog) -> None:
    """Apply consistent visibility and screen-relative sizing to a dialog."""
    # Native Linux dialogs often ignore QDir.Hidden off; Qt dialog respects it
    dlg.setOption(QFileDialog.DontUseNativeDialog, True)
    dlg.setFilter(_VISIBLE_ENTRIES)

    screen = QApplication.primaryScreen()
    if screen is not None:
        available = screen.availableGeometry()
        width = max(640, int(available.width() * 0.75))
        height = max(480, int(available.height() * 0.75))
        dlg.resize(width, height)
        dlg.move(
            available.x() + (available.width() - width) // 2,
            available.y() + (available.height() - height) // 2,
        )
    QTimer.singleShot(0, lambda dlg=dlg: _widen_sidebar(dlg))


def _widen_sidebar(dlg: QFileDialog) -> None:
    """Give Qt's left sidebar a readable width after the dialog shows."""
    splitter = dlg.findChild(QSplitter, "splitter")
    sidebar = dlg.findChild(QListView, "sidebar")
    if splitter is None or sidebar is None:
        return

    # Qt often restores a stale narrow splitter ratio; bump the sidebar first.
    sidebar.setMinimumWidth(180)
    sidebar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
    total_width = max(splitter.width(), dlg.width())
    left_width = max(180, int(total_width * 0.25))
    right_width = max(1, total_width - left_width)
    splitter.setSizes([left_width, right_width])


def get_open_file_name(
    parent: Optional[QWidget],
    caption: str,
    directory: str = "",
    filter: str = "All Files (*)",
) -> Tuple[str, str]:
    """Pick one existing file; hidden files/folders not shown by default.

    Returns:
        (path, selected_filter) — path is "" if cancelled.
    """
    filter = _all_files_first(filter)
    dlg = QFileDialog(parent, caption, directory, filter)
    dlg.setFileMode(QFileDialog.ExistingFile)
    dlg.setAcceptMode(QFileDialog.AcceptOpen)
    dlg.setNameFilter(filter)
    _prepare_dialog(dlg)
    if dlg.exec_():
        files = dlg.selectedFiles()
        return (files[0] if files else "", dlg.selectedNameFilter())
    return "", ""


def get_open_file_names(
    parent: Optional[QWidget],
    caption: str,
    directory: str = "",
    filter: str = "All Files (*)",
) -> Tuple[List[str], str]:
    """Pick multiple existing files; hidden files/folders not shown by default.

    Returns:
        (paths, selected_filter) — paths is [] if cancelled.
    """
    filter = _all_files_first(filter)
    dlg = QFileDialog(parent, caption, directory, filter)
    dlg.setFileMode(QFileDialog.ExistingFiles)
    dlg.setAcceptMode(QFileDialog.AcceptOpen)
    dlg.setNameFilter(filter)
    _prepare_dialog(dlg)
    if dlg.exec_():
        return list(dlg.selectedFiles()), dlg.selectedNameFilter()
    return [], ""


def get_existing_directory(
    parent: Optional[QWidget],
    caption: str,
    directory: str = "",
) -> str:
    """Pick an existing directory; hidden folders not shown by default.

    Returns:
        Selected path, or "" if cancelled.
    """
    dlg = QFileDialog(parent, caption, directory)
    dlg.setFileMode(QFileDialog.Directory)
    dlg.setOption(QFileDialog.ShowDirsOnly, True)
    _prepare_dialog(dlg)
    if dlg.exec_():
        files = dlg.selectedFiles()
        return files[0] if files else ""
    return ""
