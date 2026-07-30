"""Desktop interface for DECIMER image-to-SMILES recognition."""

from __future__ import annotations

import csv
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QFont, QFontDatabase
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"
}


@dataclass(frozen=True)
class Job:
    image: Path
    relative_name: str


class RecognitionWorker(QObject):
    model_ready = Signal()
    stage_changed = Signal(str)
    result_done = Signal(str, str, str, str, str)
    progress = Signal(int, int)
    finished = Signal(str, int, int, bool)
    fatal_error = Signal(str)

    def __init__(
        self, jobs: list[Job], output_dir: Path, hand_drawn: bool, segmentation: bool
    ):
        super().__init__()
        self.jobs = jobs
        self.output_dir = output_dir
        self.hand_drawn = hand_drawn
        self.segmentation = segmentation
        self._cancelled = False

    @Slot()
    def cancel(self):
        self._cancelled = True

    @Slot()
    def run(self):
        try:
            # Importing DECIMER downloads (on first run) and loads the models.
            from DECIMER import predict_SMILES
            import cv2

            segment_from_file = None
            if self.segmentation:
                self.stage_changed.emit("正在加载分割模型…首次运行需要下载模型")
                from decimer_segmentation import segment_chemical_structures_from_file

                segment_from_file = segment_chemical_structures_from_file

            self.model_ready.emit()
            self.output_dir.mkdir(parents=True, exist_ok=True)
            results: list[tuple[str, str, str, str, str]] = []
            completed = 0
            failed = 0

            for source_index, job in enumerate(self.jobs):
                if self._cancelled:
                    break
                candidates: list[tuple[str, object, Path | None]] = []

                if self.segmentation:
                    self.stage_changed.emit(
                        f"正在分割 {source_index + 1}/{len(self.jobs)}：{job.image.name}"
                    )
                    try:
                        segments = segment_from_file(str(job.image), expand=True)
                        segment_dir = (
                            self.output_dir / "segments" / Path(job.relative_name).with_suffix("")
                        )
                        segment_dir.mkdir(parents=True, exist_ok=True)
                        for segment_index, segment in enumerate(segments, start=1):
                            segment_name = f"structure_{segment_index:03d}"
                            crop_path = segment_dir / f"{segment_name}.png"
                            if not cv2.imwrite(str(crop_path), segment):
                                raise OSError(f"无法保存分割图片：{crop_path}")
                            candidates.append((segment_name, segment, crop_path))
                        if not candidates:
                            status = "未检测到化学结构"
                            results.append((job.relative_name, "", "", "", status))
                            self.result_done.emit(job.relative_name, "", "", "", status)
                    except Exception as exc:
                        status = f"分割失败：{exc}"
                        results.append((job.relative_name, "", "", "", status))
                        self.result_done.emit(job.relative_name, "", "", "", status)
                        failed += 1
                else:
                    candidates.append(("整张图片", str(job.image), None))

                for segment_name, image_input, crop_path in candidates:
                    if self._cancelled:
                        break
                    self.stage_changed.emit(
                        f"正在识别：{job.image.name} / {segment_name}"
                    )
                    try:
                        smiles = str(
                            predict_SMILES(image_input, hand_drawn=self.hand_drawn)
                        ).strip()
                        if crop_path:
                            result_file = crop_path.with_suffix(".smi")
                        else:
                            result_file = (
                                self.output_dir / Path(job.relative_name).with_suffix(".smi")
                            )
                        result_file.parent.mkdir(parents=True, exist_ok=True)
                        result_file.write_text(smiles + "\n", encoding="utf-8")
                        status = "已完成"
                        completed += 1
                    except Exception as exc:  # Keep the batch running after one bad image.
                        smiles = ""
                        status = f"识别失败：{exc}"
                        failed += 1
                    crop_text = str(crop_path) if crop_path else ""
                    results.append(
                        (job.relative_name, segment_name, crop_text, smiles, status)
                    )
                    self.result_done.emit(
                        job.relative_name, segment_name, crop_text, smiles, status
                    )
                self.progress.emit(source_index + 1, len(self.jobs))

            summary = self.output_dir / "smiles_results.csv"
            try:
                self._write_summary(summary, results)
            except PermissionError:
                # Excel commonly locks an opened CSV on Windows. Preserve the
                # completed recognition results in a new file instead of
                # treating the whole batch as failed.
                summary = self.output_dir / (
                    f"smiles_results_{datetime.now():%Y%m%d_%H%M%S}.csv"
                )
                self._write_summary(summary, results)

            self.finished.emit(str(summary), completed, failed, self._cancelled)
        except Exception:
            self.fatal_error.emit(traceback.format_exc())

    @staticmethod
    def _write_summary(
        summary: Path, results: list[tuple[str, str, str, str, str]]
    ):
        with summary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["source_image", "segment", "segment_image", "smiles", "status"]
            )
            writer.writerows(results)


class DropZone(QFrame):
    paths_dropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setObjectName("dropZone")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 30, 32, 30)
        title = QLabel("将化学结构图片拖到这里")
        title.setObjectName("dropTitle")
        title.setAlignment(Qt.AlignCenter)
        note = QLabel("支持 PNG、JPG、TIFF、WEBP、HEIC，也可以直接拖入文件夹")
        note.setObjectName("dropNote")
        note.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(note)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        paths = [url.toLocalFile() for url in event.mimeData().urls()]
        self.paths_dropped.emit(paths)
        event.acceptProposedAction()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DECIMER · 化学结构识别")
        self.resize(1060, 760)
        self.jobs: list[Job] = []
        self.output_dir = Path.cwd() / "DECIMER_results"
        self.worker: RecognitionWorker | None = None
        self.thread: QThread | None = None
        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(36, 30, 36, 30)
        layout.setSpacing(18)

        eyebrow = QLabel("OPTICAL CHEMICAL STRUCTURE RECOGNITION")
        eyebrow.setObjectName("eyebrow")
        title = QLabel("从结构图，得到 SMILES")
        title.setObjectName("title")
        subtitle = QLabel("单张拖入或整批处理。每张图片生成一个 .smi 文件，并附带 CSV 汇总表。")
        subtitle.setObjectName("subtitle")
        layout.addWidget(eyebrow)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        self.drop_zone = DropZone()
        self.drop_zone.paths_dropped.connect(self.add_paths)
        layout.addWidget(self.drop_zone)

        actions = QHBoxLayout()
        self.add_files_button = QPushButton("选择图片")
        self.add_folder_button = QPushButton("选择文件夹")
        self.output_button = QPushButton("保存到…")
        self.clear_button = QPushButton("清空列表")
        self.segmentation = QCheckBox("先分割页面中的结构")
        self.segmentation.setToolTip(
            "适用于论文页面、扫描页或一张图中含多个结构；单个结构图片无需勾选。"
        )
        self.hand_drawn = QCheckBox("手绘结构模型")
        self.recursive = QCheckBox("包含子文件夹")
        actions.addWidget(self.add_files_button)
        actions.addWidget(self.add_folder_button)
        actions.addWidget(self.output_button)
        actions.addWidget(self.clear_button)
        actions.addStretch()
        actions.addWidget(self.recursive)
        actions.addWidget(self.segmentation)
        actions.addWidget(self.hand_drawn)
        layout.addLayout(actions)

        self.mode_label = QLabel(
            "当前流程：整张图片 → SMILES（适合每张图只有一个独立结构）"
        )
        self.mode_label.setObjectName("modeLabel")
        self.mode_label.setWordWrap(True)
        layout.addWidget(self.mode_label)

        self.add_feedback = QLabel("尚未添加图片")
        self.add_feedback.setObjectName("addFeedback")
        self.add_feedback.setWordWrap(True)
        layout.addWidget(self.add_feedback)

        self.output_label = QLabel(f"输出文件夹：{self.output_dir}")
        self.output_label.setObjectName("pathLabel")
        layout.addWidget(self.output_label)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["来源图片", "分割结构", "分割图片", "SMILES", "状态"]
        )
        self.table.setMinimumHeight(210)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        layout.addWidget(self.table, 1)

        footer = QHBoxLayout()
        self.status_label = QLabel("等待添加图片")
        self.status_label.setObjectName("statusLabel")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.run_button = QPushButton("开始识别")
        self.run_button.setObjectName("primary")
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setEnabled(False)
        footer.addWidget(self.status_label)
        footer.addWidget(self.progress, 1)
        footer.addWidget(self.cancel_button)
        footer.addWidget(self.run_button)
        layout.addLayout(footer)

        self.add_files_button.clicked.connect(self.choose_files)
        self.add_folder_button.clicked.connect(self.choose_folder)
        self.output_button.clicked.connect(self.choose_output)
        self.clear_button.clicked.connect(self.clear_jobs)
        self.run_button.clicked.connect(self.start_recognition)
        self.cancel_button.clicked.connect(self.cancel_recognition)
        self.segmentation.toggled.connect(self.on_segmentation_toggled)

    @Slot(bool)
    def on_segmentation_toggled(self, enabled: bool):
        if enabled:
            self.mode_label.setText(
                "当前流程：页面图片 → 分割多个化学结构 → 逐个识别 → SMILES"
            )
        else:
            self.mode_label.setText(
                "当前流程：整张图片 → SMILES（适合每张图只有一个独立结构）"
            )

    def _apply_style(self):
        self.setStyleSheet("""
            QWidget#root { background: #F3F7F6; color: #17332F; }
            QLabel#eyebrow { color: #287F71; font-size: 11px; font-weight: 700; letter-spacing: 2px; }
            QLabel#title { color: #102E29; font-size: 30px; font-weight: 700; }
            QLabel#subtitle { color: #58706B; font-size: 14px; }
            QFrame#dropZone {
                background: #E3F0ED; border: 2px dashed #72A99F; border-radius: 16px;
            }
            QFrame#dropZone:hover { background: #D9EBE7; border-color: #287F71; }
            QLabel#dropTitle { color: #173E37; font-size: 18px; font-weight: 700; }
            QLabel#dropNote, QLabel#pathLabel { color: #68807B; font-size: 12px; }
            QLabel#modeLabel {
                color: #176B5D; background: #DDEEEA; border-radius: 6px;
                padding: 7px 10px; font-size: 12px;
            }
            QLabel#addFeedback {
                background: #FFFFFF; border-left: 4px solid #E4863A; border-radius: 5px;
                color: #365C55; padding: 8px 12px; font-size: 12px; font-weight: 600;
            }
            QPushButton {
                background: #FFFFFF; border: 1px solid #C7D8D4; border-radius: 8px;
                padding: 8px 14px; color: #244A43; font-size: 13px;
            }
            QPushButton:hover { border-color: #287F71; color: #176B5D; }
            QPushButton:disabled { color: #9AA9A6; background: #EDF1F0; }
            QPushButton#primary {
                background: #176B5D; color: white; border: none; padding: 10px 24px;
                font-weight: 700;
            }
            QPushButton#primary:hover { background: #10584D; }
            QCheckBox { spacing: 7px; color: #365C55; font-size: 12px; }
            QTableWidget {
                background: white; border: 1px solid #D8E3E0; border-radius: 10px;
                gridline-color: #E8EFED; selection-background-color: #DCEDE9;
                selection-color: #17332F; color: #17332F;
                font-size: 12px;
            }
            QTableWidget::item {
                color: #17332F; background: #FFFFFF; padding: 6px;
            }
            QTableWidget::item:selected {
                color: #102E29; background: #DCEDE9;
            }
            QHeaderView::section {
                background: #EDF4F2; color: #365C55; border: none;
                border-bottom: 1px solid #D8E3E0; padding: 9px; font-weight: 700;
            }
            QProgressBar { background: #DCE7E4; border: none; border-radius: 4px; height: 8px; }
            QProgressBar::chunk { background: #E4863A; border-radius: 4px; }
            QLabel#statusLabel { color: #486760; min-width: 190px; }
        """)

    @Slot()
    def choose_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择化学结构图片", "", "图片 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.heic *.heif)"
        )
        self.add_paths(files)

    @Slot()
    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder:
            self.add_paths([folder])

    @Slot()
    def choose_output(self):
        folder = QFileDialog.getExistingDirectory(self, "选择结果保存文件夹")
        if folder:
            self.output_dir = Path(folder)
            self.output_label.setText(f"输出文件夹：{self.output_dir}")

    @Slot(list)
    def add_paths(self, paths: list[str]):
        existing = {job.image.resolve() for job in self.jobs}
        added: list[Job] = []
        ignored = 0
        for raw in paths:
            path = Path(raw)
            if path.is_dir():
                iterator = path.rglob("*") if self.recursive.isChecked() else path.glob("*")
                found_supported = False
                for image in sorted(iterator):
                    if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
                        found_supported = True
                        resolved = image.resolve()
                        if resolved not in existing:
                            added.append(Job(resolved, str(image.relative_to(path))))
                            existing.add(resolved)
                        else:
                            ignored += 1
                if not found_supported:
                    ignored += 1
            elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                resolved = path.resolve()
                if resolved not in existing:
                    added.append(Job(resolved, path.name))
                    existing.add(resolved)
                else:
                    ignored += 1
            else:
                ignored += 1
        self.jobs.extend(added)
        self._refresh_table()
        if added:
            self.table.selectRow(len(self.jobs) - 1)
            self.table.scrollToBottom()
            names = "、".join(job.image.name for job in added[:3])
            more = f" 等 {len(added)} 张" if len(added) > 3 else ""
            ignored_text = f"；忽略 {ignored} 项（重复或格式不支持）" if ignored else ""
            self.add_feedback.setText(
                f"✓ 本次已添加：{names}{more}；列表共 {len(self.jobs)} 张{ignored_text}"
            )
        else:
            self.add_feedback.setText(
                "未添加新图片：文件可能已经在列表中，或格式不受支持。"
            )
        self.status_label.setText(f"已添加 {len(self.jobs)} 张图片，等待识别")

    def _refresh_table(self):
        self.table.setRowCount(len(self.jobs))
        for row, job in enumerate(self.jobs):
            values = (job.image.name, "等待分割/识别", "", "", "等待处理")
            for column, value in enumerate(values):
                item = QTableWidgetItem()
                item.setText(value)
                item.setForeground(QColor("#17332F"))
                item.setBackground(QColor("#FFFFFF"))
                item.setFont(QApplication.font())
                item.setToolTip(str(job.image))
                self.table.setItem(row, column, item)
            self.table.setRowHeight(row, 38)
        self.table.viewport().update()

    @Slot()
    def clear_jobs(self):
        self.jobs.clear()
        self.table.setRowCount(0)
        self.add_feedback.setText("尚未添加图片")
        self.status_label.setText("等待添加图片")

    @Slot()
    def start_recognition(self):
        if not self.jobs:
            QMessageBox.information(self, "没有图片", "请先添加至少一张化学结构图片。")
            return
        self.set_controls_enabled(False)
        self.progress.setRange(0, len(self.jobs))
        self.progress.setValue(0)
        self.status_label.setText("正在加载模型…首次运行需要下载模型")

        self.thread = QThread(self)
        self.worker = RecognitionWorker(
            list(self.jobs),
            self.output_dir,
            self.hand_drawn.isChecked(),
            self.segmentation.isChecked(),
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.model_ready.connect(lambda: self.status_label.setText("模型已就绪，正在识别…"))
        self.worker.stage_changed.connect(self.status_label.setText)
        self.worker.result_done.connect(self.on_result_done)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.fatal_error.connect(self.on_fatal_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.fatal_error.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    @Slot()
    def cancel_recognition(self):
        if self.worker:
            self.worker.cancel()
            self.status_label.setText("正在完成当前图片后取消…")

    @Slot(str, str, str, str, str)
    def on_result_done(
        self, source: str, segment: str, crop_path: str, smiles: str, status: str
    ):
        # Replace queued source rows with actual direct/segmented recognition results.
        if self.table.rowCount() == len(self.jobs):
            queued = all(
                self.table.item(row, 4)
                and self.table.item(row, 4).text() == "等待处理"
                for row in range(self.table.rowCount())
            )
            if queued:
                self.table.setRowCount(0)
        row = self.table.rowCount()
        self.table.insertRow(row)
        values = (
            source,
            segment or "—",
            crop_path or "—",
            smiles,
            status,
        )
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setForeground(
                QColor("#B34C35")
                if ("失败" in status or "未检测到" in status)
                else QColor("#17332F")
            )
            item.setToolTip(value)
            self.table.setItem(row, column, item)
        self.table.setRowHeight(row, 38)
        self.table.scrollToBottom()

    @Slot(int, int)
    def on_progress(self, current: int, total: int):
        self.progress.setValue(current)
        self.status_label.setText(f"正在识别 {current}/{total}")

    @Slot(str, int, int, bool)
    def on_finished(self, summary: str, completed: int, failed: int, cancelled: bool):
        self.set_controls_enabled(True)
        state = "已取消" if cancelled else "识别完成"
        self.status_label.setText(f"{state}：成功 {completed}，失败 {failed}")
        QMessageBox.information(
            self, state, f"成功：{completed}\n失败：{failed}\n\n汇总结果：\n{summary}"
        )
        self.worker = None
        self.thread = None

    @Slot(str)
    def on_fatal_error(self, details: str):
        self.set_controls_enabled(True)
        self.status_label.setText("模型加载或识别失败")
        log_file = self.output_dir / f"error_{datetime.now():%Y%m%d_%H%M%S}.log"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_file.write_text(details, encoding="utf-8")
        QMessageBox.critical(
            self, "无法完成识别",
            f"错误详情已保存到：\n{log_file}\n\n请检查网络连接和 Python 依赖。"
        )
        self.worker = None
        self.thread = None

    def set_controls_enabled(self, enabled: bool):
        for control in (
            self.add_files_button, self.add_folder_button, self.output_button,
            self.clear_button, self.run_button, self.hand_drawn, self.segmentation,
            self.recursive
        ):
            control.setEnabled(enabled)
        self.cancel_button.setEnabled(not enabled)
        self.drop_zone.setAcceptDrops(enabled)


def main():
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication(sys.argv)
    app.setApplicationName("DECIMER Desktop")
    # Loading the font file explicitly avoids missing-glyph boxes in some Qt
    # installations where the Windows font fallback chain is not discovered.
    preferred_font = None
    for font_path in (
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "msyh.ttc",
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "segoeui.ttf",
    ):
        if font_path.exists():
            font_id = QFontDatabase.addApplicationFont(str(font_path))
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                preferred_font = QFont(families[0], 10)
                break
    if preferred_font is None:
        preferred_font = QFontDatabase.systemFont(QFontDatabase.GeneralFont)
        preferred_font.setPointSize(10)
    app.setFont(preferred_font)

    # Non-interactive smoke test used to verify packaged builds. It is only
    # activated by an explicit environment variable and has no effect on
    # normal desktop launches.
    self_test_input = os.environ.get("DECIMER_SELF_TEST_INPUT")
    if self_test_input:
        image_path = Path(self_test_input).resolve()
        output_path = Path(
            os.environ.get("DECIMER_SELF_TEST_OUTPUT", "package_self_test")
        ).resolve()
        errors: list[str] = []
        worker = RecognitionWorker(
            [Job(image_path, image_path.name)],
            output_path,
            hand_drawn=False,
            segmentation=os.environ.get("DECIMER_SELF_TEST_SEGMENTATION") == "1",
        )
        worker.fatal_error.connect(errors.append)
        worker.run()
        if errors:
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / "fatal_error.log").write_text(errors[0], encoding="utf-8")
            sys.exit(1)
        sys.exit(0)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
