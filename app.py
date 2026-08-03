
from __future__ import annotations

import json
import os
import random
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import requests
from PIL import Image, ImageOps
from PySide6.QtCore import QObject, QThread, Signal, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget, QScrollArea,
    QProgressDialog
)

from portrait_engine import PortraitEngine, RenderSettings
from postprocess import PostProcessSettings, PostProcessor


# APP_DIR yazılabilir kullanıcı dosyalarının bulunduğu klasördür.
# BUNDLE_DIR ise PyInstaller'ın paketlediği salt-okunur kaynakların konumudur.
APP_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent
BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
CONFIG_PATH = APP_DIR / "config.json"


def load_config() -> dict:
    example = BUNDLE_DIR / "config.example.json"
    if not example.exists():
        # Kaynak koddan çalıştırma veya elle kopyalanmış kurulum için geri dönüş.
        example = APP_DIR / "config.example.json"

    if not CONFIG_PATH.exists():
        if not example.exists():
            raise FileNotFoundError(
                "config.example.json bulunamadı. EXE klasörünü eksik kopyalamış olabilirsin."
            )
        CONFIG_PATH.write_text(
            example.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def pil_to_pixmap(
    img: Image.Image,
    max_side: int = 460,
    checkerboard: bool = False,
) -> QPixmap:
    img = img.copy()
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    if checkerboard and img.mode == "RGBA":
        tile = 16
        arr = np.zeros((img.height, img.width, 3), dtype=np.uint8)
        for y in range(0, img.height, tile):
            for x in range(0, img.width, tile):
                value = 205 if ((x // tile) + (y // tile)) % 2 == 0 else 155
                arr[y:y + tile, x:x + tile] = value
        bg = Image.fromarray(arr, "RGB").convert("RGBA")
        img = Image.alpha_composite(bg, img.convert("RGBA")).convert("RGB")

    if img.mode == "RGBA":
        data = img.tobytes("raw", "RGBA")
        qimg = QImage(data, img.width, img.height, QImage.Format_RGBA8888)
    else:
        img = img.convert("RGB")
        data = img.tobytes("raw", "RGB")
        qimg = QImage(data, img.width, img.height, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def download_image(url: str, timeout: int = 30) -> Image.Image:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FootballPortraitStudio/1.0",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    from io import BytesIO
    return Image.open(BytesIO(response.content)).convert("RGB")


class Worker(QObject):
    progress = Signal(int, int, str)
    preview = Signal(object, object, str, str)
    log = Signal(str)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, job: dict):
        super().__init__()
        self.job = job
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def run(self):
        stats = {"completed": 0, "failed": 0, "skipped": 0}
        report = []
        try:
            cfg = self.job["config"]
            settings = RenderSettings(**self.job["settings"])
            engine = PortraitEngine(
                base_model=cfg["base_model"],
                ip_adapter_repo=cfg["ip_adapter_repo"],
                ip_adapter_subfolder=cfg["ip_adapter_subfolder"],
                ip_adapter_weight=cfg["ip_adapter_weight"],
                lora_path=cfg.get("lora_path", ""),
                lora_trigger=cfg.get("lora_trigger", ""),
                status_cb=self.log.emit,
            )

            df = pd.read_csv(self.job["csv_path"])
            id_col = self.job["id_col"]
            name_col = self.job["name_col"]
            url_col = self.job["url_col"]

            start = self.job["start_index"]
            limit = self.job["limit"]
            selected = df.iloc[start:] if limit == 0 else df.iloc[start:start + limit]

            output_dir = Path(self.job["output_dir"])
            completed_dir = output_dir / "completed"
            failed_dir = output_dir / "failed"
            source_dir = output_dir / "sources"
            for d in (completed_dir, failed_dir, source_dir):
                d.mkdir(parents=True, exist_ok=True)

            total = len(selected)
            for n, (_, row) in enumerate(selected.iterrows(), start=1):
                if self.cancelled:
                    self.log.emit("İşlem kullanıcı tarafından durduruldu.")
                    break

                player_id = str(row[id_col]).strip()
                name = str(row.get(name_col, player_id)).strip()
                url = str(row[url_col]).strip()
                output_path = completed_dir / f"{player_id}.png"

                if self.job["skip_existing"] and output_path.exists():
                    stats["skipped"] += 1
                    self.progress.emit(n, total, f"Atlandı: {name}")
                    continue

                try:
                    self.progress.emit(n, total, f"İndiriliyor: {name}")
                    source = download_image(url)
                    source.save(source_dir / f"{player_id}.jpg", quality=95)

                    local_settings = RenderSettings(**self.job["settings"])
                    if self.job["random_seed"]:
                        local_settings.seed = random.randint(0, 2_147_483_647)

                    self.progress.emit(n, total, f"Karikatürize ediliyor: {name}")
                    result = engine.render(source, local_settings)
                    result.save(output_path)

                    stats["completed"] += 1
                    self.preview.emit(source, result, name, str(output_path))
                    report.append({
                        "player_id": player_id,
                        "name": name,
                        "source_url": url,
                        "status": "completed",
                        "seed": local_settings.seed,
                        "output_path": str(output_path),
                        "error": "",
                    })
                except Exception as exc:
                    stats["failed"] += 1
                    err = f"{type(exc).__name__}: {exc}"
                    self.log.emit(f"HATA — {name}: {err}")
                    report.append({
                        "player_id": player_id,
                        "name": name,
                        "source_url": url,
                        "status": "failed",
                        "seed": self.job["settings"]["seed"],
                        "output_path": "",
                        "error": err,
                    })

                self.progress.emit(n, total, name)

            pd.DataFrame(report).to_csv(
                output_dir / "report.csv", index=False, encoding="utf-8-sig"
            )
            self.finished.emit(stats)
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Football Portrait Studio v1.3.3")
        self.resize(1280, 820)
        self.config = load_config()
        self.worker: Optional[Worker] = None
        self.thread: Optional[QThread] = None

        self.post_processor = PostProcessor()
        self.post_raw_image: Optional[Image.Image] = None
        self.post_result_image: Optional[Image.Image] = None
        self.post_source_path: Optional[Path] = None
        self.post_display_name: str = ""
        self.post_timer = QTimer(self)
        self.post_timer.setSingleShot(True)
        self.post_timer.setInterval(220)
        self.post_timer.timeout.connect(self.update_post_preview)

        tabs = QTabWidget()
        tabs.addTab(self._build_batch_tab(), "Toplu Üretim")
        tabs.addTab(self._build_postprocess_tab(), "Son İşlem")
        tabs.addTab(self._build_model_tab(), "Model Ayarları")
        tabs.addTab(self._build_help_tab(), "Kısa Yardım")
        self.tabs = tabs
        self.setCentralWidget(tabs)

    def _build_batch_tab(self) -> QWidget:
        root = QWidget()
        layout = QHBoxLayout(root)
        splitter = QSplitter(Qt.Horizontal)

        controls = QWidget()
        c = QVBoxLayout(controls)

        files_box = QGroupBox("Dosyalar")
        form = QFormLayout(files_box)
        self.csv_edit = QLineEdit()
        self.out_edit = QLineEdit(str(APP_DIR / "output"))
        self.lora_edit = QLineEdit(self.config.get("lora_path", ""))

        form.addRow("CSV:", self._path_row(self.csv_edit, self.pick_csv))
        form.addRow("Çıktı klasörü:", self._path_row(self.out_edit, self.pick_output))
        form.addRow("Stil LoRA:", self._path_row(self.lora_edit, self.pick_lora))
        c.addWidget(files_box)

        columns_box = QGroupBox("CSV sütunları")
        form2 = QFormLayout(columns_box)
        self.id_col = QLineEdit("player_id")
        self.name_col = QLineEdit("name")
        self.url_col = QLineEdit("image_url")
        form2.addRow("ID:", self.id_col)
        form2.addRow("İsim:", self.name_col)
        form2.addRow("Fotoğraf URL:", self.url_col)
        c.addWidget(columns_box)

        settings_box = QGroupBox("Üretim kontrolleri")
        grid = QGridLayout(settings_box)
        self.face = self._double(0.76, 0.0, 1.0, 0.01)
        self.caricature = self._double(0.68, 0.05, 0.90, 0.01)
        self.style = self._double(0.90, 0.0, 1.5, 0.05)
        self.guidance = self._double(7.0, 1.0, 15.0, 0.5)
        self.steps = self._spin(24, 8, 60)
        self.seed = self._spin(12345, 0, 2_147_483_647)
        self.size = QComboBox()
        self.size.addItems(["384", "512", "640", "768"])
        self.size.setCurrentText(str(self.config.get("output_size", 512)))
        self.sharpen = self._double(1.15, 0.0, 3.0, 0.05)
        self.colors = self._double(0.12, 0.0, 1.0, 0.05)

        labels_widgets = [
            ("Yüz/kimlik gücü", self.face),
            ("Karikatür / denoise", self.caricature),
            ("LoRA stil gücü", self.style),
            ("Prompt guidance", self.guidance),
            ("Adım sayısı", self.steps),
            ("Seed", self.seed),
            ("Çıktı boyutu", self.size),
            ("Keskinlik", self.sharpen),
            ("Kaynak renk karışımı", self.colors),
        ]
        for i, (label, widget) in enumerate(labels_widgets):
            grid.addWidget(QLabel(label), i // 2, (i % 2) * 2)
            grid.addWidget(widget, i // 2, (i % 2) * 2 + 1)

        c.addWidget(settings_box)

        range_box = QGroupBox("Aralık ve seçenekler")
        form3 = QFormLayout(range_box)
        self.start_index = self._spin(0, 0, 10_000_000)
        self.limit = self._spin(3, 0, 10_000_000)
        self.limit.setSpecialValueText("Tümü")
        self.skip_existing = QCheckBox("Var olan PNG'leri atla")
        self.skip_existing.setChecked(True)
        self.random_seed = QCheckBox("Her oyuncuya farklı seed")
        self.random_seed.setChecked(True)
        self.remove_bg = QCheckBox("Üretimde arka plan kaldır (eski yöntem)")
        self.remove_bg.setChecked(False)
        self.use_ip = QCheckBox("IP-Adapter yüz/kimlik rehberi")
        self.use_ip.setChecked(True)
        self.use_lora = QCheckBox("Stil LoRA kullan")
        self.use_lora.setChecked(True)
        form3.addRow("Başlangıç satırı:", self.start_index)
        form3.addRow("Kaç oyuncu:", self.limit)
        form3.addRow(self.skip_existing)
        form3.addRow(self.random_seed)
        form3.addRow(self.remove_bg)
        form3.addRow(self.use_ip)
        form3.addRow(self.use_lora)
        c.addWidget(range_box)

        prompt_box = QGroupBox("Prompt")
        pv = QVBoxLayout(prompt_box)
        self.prompt = QPlainTextEdit(self.config.get("default_prompt", ""))
        self.negative = QPlainTextEdit(self.config.get("negative_prompt", ""))
        self.prompt.setMaximumHeight(90)
        self.negative.setMaximumHeight(80)
        pv.addWidget(QLabel("Pozu değiştirmemesi özellikle promptta belirtilmiştir."))
        pv.addWidget(self.prompt)
        pv.addWidget(QLabel("Negative prompt"))
        pv.addWidget(self.negative)
        c.addWidget(prompt_box)

        preset_box = QGroupBox("Hızlı ayar")
        preset_row = QHBoxLayout(preset_box)
        identity_btn = QPushButton("Kimlik ağırlıklı")
        balanced_btn = QPushButton("Dengeli")
        style_btn = QPushButton("Karikatür güçlü")
        identity_btn.clicked.connect(lambda: self.apply_preset("identity"))
        balanced_btn.clicked.connect(lambda: self.apply_preset("balanced"))
        style_btn.clicked.connect(lambda: self.apply_preset("style"))
        preset_row.addWidget(identity_btn)
        preset_row.addWidget(balanced_btn)
        preset_row.addWidget(style_btn)
        c.addWidget(preset_box)

        buttons = QHBoxLayout()
        self.start_btn = QPushButton("Üretimi Başlat")
        self.stop_btn = QPushButton("Durdur")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_job)
        self.stop_btn.clicked.connect(self.stop_job)
        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.stop_btn)
        c.addLayout(buttons)

        self.progress = QProgressBar()
        self.status = QLabel("Hazır")
        c.addWidget(self.progress)
        c.addWidget(self.status)
        c.addStretch(1)

        right = QWidget()
        r = QVBoxLayout(right)
        previews = QHBoxLayout()
        self.source_preview = QLabel("Kaynak")
        self.result_preview = QLabel("Sonuç")
        for lab in (self.source_preview, self.result_preview):
            lab.setAlignment(Qt.AlignCenter)
            lab.setMinimumSize(380, 380)
            lab.setStyleSheet("QLabel { border: 1px solid #555; background:#111; color:#aaa; }")
        previews.addWidget(self.source_preview)
        previews.addWidget(self.result_preview)
        r.addLayout(previews)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        r.addWidget(self.log)

        splitter.addWidget(controls)
        splitter.addWidget(right)
        splitter.setSizes([500, 780])
        layout.addWidget(splitter)
        return root


    def _build_postprocess_tab(self) -> QWidget:
        root = QWidget()
        layout = QHBoxLayout(root)
        splitter = QSplitter(Qt.Horizontal)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        controls = QWidget()
        c = QVBoxLayout(controls)

        file_box = QGroupBox("Görsel")
        file_form = QFormLayout(file_box)
        self.post_file_label = QLabel("Henüz görsel yok")
        self.post_file_label.setWordWrap(True)
        open_btn = QPushButton("PNG / JPG Aç")
        open_btn.clicked.connect(self.open_post_image)
        use_last_btn = QPushButton("Son üretileni kullan")
        use_last_btn.clicked.connect(self.use_last_generated)
        file_buttons = QHBoxLayout()
        file_buttons.addWidget(open_btn)
        file_buttons.addWidget(use_last_btn)
        file_form.addRow("Aktif görsel:", self.post_file_label)
        file_form.addRow(file_buttons)
        c.addWidget(file_box)

        bg_box = QGroupBox("Arka plan kaldırma")
        bg_form = QFormLayout(bg_box)
        self.pp_bg_mode = QComboBox()
        self.pp_bg_mode.addItem("Kapalı", "off")
        self.pp_bg_mode.addItem("AI — ISNet (önerilen)", "isnet")
        self.pp_bg_mode.addItem("AI — U2Net", "u2net")
        self.pp_bg_mode.addItem("Kenar rengini temizle", "border")
        self.pp_bg_mode.addItem("Beyaz zemini temizle", "white")
        self.pp_bg_tolerance = self._spin(28, 0, 100)
        self.pp_feather = self._spin(2, 0, 12)
        self.pp_expand = self._spin(0, -8, 12)
        self.pp_dehalo = self._spin(1, 0, 6)
        self.pp_keep_largest = QCheckBox("Yalnızca en büyük kişiyi koru")
        self.pp_keep_largest.setChecked(True)
        self.pp_fill_holes = QCheckBox("Maskenin iç boşluklarını doldur")
        self.pp_fill_holes.setChecked(True)
        bg_form.addRow("Yöntem:", self.pp_bg_mode)
        bg_form.addRow("Renk toleransı:", self.pp_bg_tolerance)
        bg_form.addRow("Kenar yumuşatma:", self.pp_feather)
        bg_form.addRow("Maskeyi büyüt / küçült:", self.pp_expand)
        bg_form.addRow("Beyaz/siyah saçak temizleme:", self.pp_dehalo)
        bg_form.addRow(self.pp_keep_largest)
        bg_form.addRow(self.pp_fill_holes)
        c.addWidget(bg_box)

        color_box = QGroupBox("Renk ve ayrıntı")
        color_grid = QGridLayout(color_box)
        self.pp_brightness = self._double(1.00, 0.20, 2.00, 0.05)
        self.pp_contrast = self._double(1.00, 0.20, 2.50, 0.05)
        self.pp_saturation = self._double(1.00, 0.00, 3.00, 0.05)
        self.pp_sharpness = self._double(1.00, 0.00, 5.00, 0.10)
        self.pp_detail = self._double(0.00, 0.00, 3.00, 0.10)
        self.pp_hue = self._spin(0, -180, 180)
        self.pp_gamma = self._double(1.00, 0.20, 3.00, 0.05)
        items = [
            ("Parlaklık", self.pp_brightness),
            ("Kontrast", self.pp_contrast),
            ("Doygunluk", self.pp_saturation),
            ("Keskinlik", self.pp_sharpness),
            ("Ayrıntı", self.pp_detail),
            ("Renk tonu (°)", self.pp_hue),
            ("Gamma", self.pp_gamma),
        ]
        for i, (label, widget) in enumerate(items):
            color_grid.addWidget(QLabel(label), i // 2, (i % 2) * 2)
            color_grid.addWidget(widget, i // 2, (i % 2) * 2 + 1)
        c.addWidget(color_box)

        outline_box = QGroupBox("Kontur")
        outline_form = QFormLayout(outline_box)
        self.pp_outline = QCheckBox("Siyah dış kontur ekle")
        self.pp_outline_thickness = self._spin(4, 1, 20)
        self.pp_outline_opacity = self._double(1.00, 0.00, 1.00, 0.05)
        self.pp_outline_softness = self._spin(0, 0, 8)
        self.pp_internal_lines = self._double(0.00, 0.00, 1.00, 0.05)
        outline_form.addRow(self.pp_outline)
        outline_form.addRow("Kalınlık:", self.pp_outline_thickness)
        outline_form.addRow("Yoğunluk:", self.pp_outline_opacity)
        outline_form.addRow("Yumuşaklık:", self.pp_outline_softness)
        outline_form.addRow("İç çizgi güçlendirme:", self.pp_internal_lines)
        c.addWidget(outline_box)

        preset_box = QGroupBox("Hazır son işlem ayarları")
        preset_row = QGridLayout(preset_box)
        pp_natural = QPushButton("Doğal")
        pp_game = QPushButton("Game Sticker")
        pp_outline = QPushButton("Güçlü Kontur")
        pp_clean = QPushButton("Şeffaf Temiz")
        pp_natural.clicked.connect(lambda: self.apply_post_preset("natural"))
        pp_game.clicked.connect(lambda: self.apply_post_preset("game"))
        pp_outline.clicked.connect(lambda: self.apply_post_preset("outline"))
        pp_clean.clicked.connect(lambda: self.apply_post_preset("clean"))
        preset_row.addWidget(pp_natural, 0, 0)
        preset_row.addWidget(pp_game, 0, 1)
        preset_row.addWidget(pp_outline, 1, 0)
        preset_row.addWidget(pp_clean, 1, 1)
        c.addWidget(preset_box)

        self.pp_auto_preview = QCheckBox("Ayar değişince önizlemeyi otomatik güncelle")
        self.pp_auto_preview.setChecked(True)
        c.addWidget(self.pp_auto_preview)

        action_grid = QGridLayout()
        update_btn = QPushButton("Önizlemeyi Güncelle")
        reset_btn = QPushButton("Ayarları Sıfırla")
        save_btn = QPushButton("Postprocessed PNG Kaydet")
        save_as_btn = QPushButton("Farklı Kaydet")
        batch_btn = QPushButton("Klasöre Toplu Uygula")
        update_btn.clicked.connect(self.update_post_preview)
        reset_btn.clicked.connect(lambda: self.apply_post_preset("natural"))
        save_btn.clicked.connect(self.save_post_result)
        save_as_btn.clicked.connect(self.save_post_result_as)
        batch_btn.clicked.connect(self.batch_postprocess_folder)
        action_grid.addWidget(update_btn, 0, 0)
        action_grid.addWidget(reset_btn, 0, 1)
        action_grid.addWidget(save_btn, 1, 0)
        action_grid.addWidget(save_as_btn, 1, 1)
        action_grid.addWidget(batch_btn, 2, 0, 1, 2)
        c.addLayout(action_grid)

        self.pp_status = QLabel(
            "AI arka plan kaldırmada ilk kullanımda model indirilebilir. "
            "Hızlı sonuç için beyaz/kenar rengi modlarını deneyebilirsin."
        )
        self.pp_status.setWordWrap(True)
        c.addWidget(self.pp_status)
        c.addStretch(1)
        scroll.setWidget(controls)

        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_row = QHBoxLayout()
        self.pp_raw_preview = QLabel("Ham model çıktısı")
        self.pp_result_preview = QLabel("Düzenlenmiş sonuç")
        for label in (self.pp_raw_preview, self.pp_result_preview):
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(400, 500)
            label.setStyleSheet(
                "QLabel { border: 1px solid #555; background:#111; color:#aaa; }"
            )
        preview_row.addWidget(self.pp_raw_preview)
        preview_row.addWidget(self.pp_result_preview)
        preview_layout.addLayout(preview_row)

        splitter.addWidget(scroll)
        splitter.addWidget(preview_widget)
        splitter.setSizes([430, 900])
        layout.addWidget(splitter)

        self._connect_post_controls()
        return root

    def _connect_post_controls(self):
        controls = [
            self.pp_bg_mode,
            self.pp_bg_tolerance,
            self.pp_feather,
            self.pp_expand,
            self.pp_dehalo,
            self.pp_keep_largest,
            self.pp_fill_holes,
            self.pp_brightness,
            self.pp_contrast,
            self.pp_saturation,
            self.pp_sharpness,
            self.pp_detail,
            self.pp_hue,
            self.pp_gamma,
            self.pp_outline,
            self.pp_outline_thickness,
            self.pp_outline_opacity,
            self.pp_outline_softness,
            self.pp_internal_lines,
        ]
        for widget in controls:
            if isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self.schedule_post_preview)
            elif isinstance(widget, QCheckBox):
                widget.toggled.connect(self.schedule_post_preview)
            else:
                widget.valueChanged.connect(self.schedule_post_preview)

    def schedule_post_preview(self, *_):
        if (
            hasattr(self, "pp_auto_preview")
            and self.pp_auto_preview.isChecked()
            and self.post_raw_image is not None
        ):
            self.post_timer.start()

    def collect_post_settings(self) -> PostProcessSettings:
        return PostProcessSettings(
            background_mode=str(self.pp_bg_mode.currentData()),
            background_tolerance=self.pp_bg_tolerance.value(),
            feather=self.pp_feather.value(),
            alpha_expand=self.pp_expand.value(),
            dehalo=self.pp_dehalo.value(),
            keep_largest=self.pp_keep_largest.isChecked(),
            fill_holes=self.pp_fill_holes.isChecked(),
            brightness=self.pp_brightness.value(),
            contrast=self.pp_contrast.value(),
            saturation=self.pp_saturation.value(),
            sharpness=self.pp_sharpness.value(),
            detail=self.pp_detail.value(),
            hue=self.pp_hue.value(),
            gamma=self.pp_gamma.value(),
            outline_enabled=self.pp_outline.isChecked(),
            outline_thickness=self.pp_outline_thickness.value(),
            outline_opacity=self.pp_outline_opacity.value(),
            outline_softness=self.pp_outline_softness.value(),
            internal_line_strength=self.pp_internal_lines.value(),
        )

    def apply_post_preset(self, preset: str):
        # Sinyal yağmurunu azaltmak için auto preview geçici olarak kapatılır.
        auto = self.pp_auto_preview.isChecked()
        self.pp_auto_preview.setChecked(False)

        if preset == "game":
            self.pp_bg_mode.setCurrentIndex(self.pp_bg_mode.findData("isnet"))
            self.pp_bg_tolerance.setValue(28)
            self.pp_feather.setValue(2)
            self.pp_expand.setValue(1)
            self.pp_dehalo.setValue(1)
            self.pp_brightness.setValue(1.02)
            self.pp_contrast.setValue(1.15)
            self.pp_saturation.setValue(1.12)
            self.pp_sharpness.setValue(1.45)
            self.pp_detail.setValue(0.45)
            self.pp_hue.setValue(0)
            self.pp_gamma.setValue(1.00)
            self.pp_outline.setChecked(True)
            self.pp_outline_thickness.setValue(4)
            self.pp_outline_opacity.setValue(0.90)
            self.pp_outline_softness.setValue(0)
            self.pp_internal_lines.setValue(0.12)
        elif preset == "outline":
            self.pp_bg_mode.setCurrentIndex(self.pp_bg_mode.findData("isnet"))
            self.pp_bg_tolerance.setValue(28)
            self.pp_feather.setValue(1)
            self.pp_expand.setValue(1)
            self.pp_dehalo.setValue(1)
            self.pp_brightness.setValue(1.00)
            self.pp_contrast.setValue(1.18)
            self.pp_saturation.setValue(1.08)
            self.pp_sharpness.setValue(1.35)
            self.pp_detail.setValue(0.55)
            self.pp_hue.setValue(0)
            self.pp_gamma.setValue(1.00)
            self.pp_outline.setChecked(True)
            self.pp_outline_thickness.setValue(7)
            self.pp_outline_opacity.setValue(1.00)
            self.pp_outline_softness.setValue(0)
            self.pp_internal_lines.setValue(0.20)
        elif preset == "clean":
            self.pp_bg_mode.setCurrentIndex(self.pp_bg_mode.findData("isnet"))
            self.pp_bg_tolerance.setValue(24)
            self.pp_feather.setValue(2)
            self.pp_expand.setValue(0)
            self.pp_dehalo.setValue(2)
            self.pp_brightness.setValue(1.00)
            self.pp_contrast.setValue(1.05)
            self.pp_saturation.setValue(1.04)
            self.pp_sharpness.setValue(1.15)
            self.pp_detail.setValue(0.15)
            self.pp_hue.setValue(0)
            self.pp_gamma.setValue(1.00)
            self.pp_outline.setChecked(False)
            self.pp_internal_lines.setValue(0.00)
        else:
            self.pp_bg_mode.setCurrentIndex(self.pp_bg_mode.findData("off"))
            self.pp_bg_tolerance.setValue(28)
            self.pp_feather.setValue(2)
            self.pp_expand.setValue(0)
            self.pp_dehalo.setValue(1)
            self.pp_brightness.setValue(1.00)
            self.pp_contrast.setValue(1.00)
            self.pp_saturation.setValue(1.00)
            self.pp_sharpness.setValue(1.00)
            self.pp_detail.setValue(0.00)
            self.pp_hue.setValue(0)
            self.pp_gamma.setValue(1.00)
            self.pp_outline.setChecked(False)
            self.pp_outline_thickness.setValue(4)
            self.pp_outline_opacity.setValue(1.00)
            self.pp_outline_softness.setValue(0)
            self.pp_internal_lines.setValue(0.00)

        self.pp_auto_preview.setChecked(auto)
        if self.post_raw_image is not None:
            self.update_post_preview()

    def _set_post_source(
        self,
        image: Image.Image,
        name: str,
        source_path: Optional[str | Path] = None,
    ):
        self.post_raw_image = ImageOps.exif_transpose(image).convert("RGBA")
        self.post_result_image = self.post_raw_image.copy()
        self.post_display_name = name
        self.post_source_path = Path(source_path) if source_path else None
        self.post_processor.clear_cache()

        path_text = str(self.post_source_path) if self.post_source_path else name
        self.post_file_label.setText(path_text)
        self.pp_raw_preview.setPixmap(
            pil_to_pixmap(self.post_raw_image, max_side=520, checkerboard=True)
        )
        self.update_post_preview()

    def use_last_generated(self):
        if self.post_raw_image is None:
            QMessageBox.information(
                self,
                "Sonuç yok",
                "Önce bir oyuncu üret veya PNG/JPG Aç düğmesiyle bir görsel seç.",
            )
            return
        self.tabs.setCurrentIndex(1)
        self.update_post_preview()

    def open_post_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Düzenlenecek görseli seç",
            "",
            "Görseller (*.png *.jpg *.jpeg *.webp)",
        )
        if not path:
            return
        try:
            image = Image.open(path)
            self._set_post_source(image, Path(path).stem, path)
        except Exception as exc:
            QMessageBox.critical(self, "Açılamadı", str(exc))

    def update_post_preview(self):
        if self.post_raw_image is None:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            settings = self.collect_post_settings()
            self.pp_status.setText("Son işlem uygulanıyor...")
            QApplication.processEvents()
            self.post_result_image = self.post_processor.process(
                self.post_raw_image,
                settings,
            )
            self.pp_result_preview.setPixmap(
                pil_to_pixmap(self.post_result_image, max_side=520, checkerboard=True)
            )
            self.pp_status.setText(
                f"Hazır — {self.post_result_image.width}×{self.post_result_image.height}, "
                f"mod: {settings.background_mode}"
            )
        except Exception as exc:
            self.pp_status.setText(f"HATA: {type(exc).__name__}: {exc}")
            QMessageBox.critical(self, "Son işlem hatası", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _default_post_output_path(self) -> Path:
        if self.post_source_path:
            source = self.post_source_path
            if source.parent.name.lower() == "completed":
                out_dir = source.parent.parent / "postprocessed"
            else:
                out_dir = source.parent / "postprocessed"
            out_dir.mkdir(parents=True, exist_ok=True)
            return out_dir / f"{source.stem}.png"

        out_dir = Path(self.out_edit.text().strip() or str(APP_DIR / "output"))
        out_dir = out_dir / "postprocessed"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = self.post_display_name or "portrait"
        return out_dir / f"{stem}.png"

    def save_post_result(self):
        if self.post_result_image is None:
            QMessageBox.information(self, "Sonuç yok", "Önce bir görsel yükle veya üret.")
            return
        path = self._default_post_output_path()
        self.post_result_image.save(path)
        self.pp_status.setText(f"Kaydedildi: {path}")
        QMessageBox.information(self, "Kaydedildi", str(path))

    def save_post_result_as(self):
        if self.post_result_image is None:
            QMessageBox.information(self, "Sonuç yok", "Önce bir görsel yükle veya üret.")
            return
        default_path = str(self._default_post_output_path())
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Düzenlenmiş PNG'yi kaydet",
            default_path,
            "PNG (*.png);;WebP (*.webp);;JPEG (*.jpg *.jpeg)",
        )
        if not path:
            return
        output = Path(path)
        if output.suffix.lower() in {".jpg", ".jpeg"}:
            # JPEG şeffaflık taşımaz; beyaz zemine birleştir.
            image = Image.new("RGB", self.post_result_image.size, "white")
            rgba = self.post_result_image.convert("RGBA")
            image.paste(rgba.convert("RGB"), mask=rgba.getchannel("A"))
            image.save(output, quality=95)
        else:
            self.post_result_image.save(output)
        self.pp_status.setText(f"Kaydedildi: {output}")

    def batch_postprocess_folder(self):
        input_dir = QFileDialog.getExistingDirectory(
            self,
            "İşlenecek görsellerin klasörü",
            str(Path(self.out_edit.text().strip() or APP_DIR / "output") / "completed"),
        )
        if not input_dir:
            return
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Çıktı klasörü",
            str(Path(input_dir).parent / "postprocessed"),
        )
        if not output_dir:
            return

        files = sorted(
            p for p in Path(input_dir).iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if not files:
            QMessageBox.information(self, "Görsel yok", "Seçilen klasörde görsel bulunamadı.")
            return

        settings = self.collect_post_settings()
        progress = QProgressDialog(
            "Son işlem uygulanıyor...",
            "İptal",
            0,
            len(files),
            self,
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        completed = 0
        errors = []
        processor = PostProcessor()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for index, path in enumerate(files, start=1):
            if progress.wasCanceled():
                break
            progress.setLabelText(f"{index}/{len(files)} — {path.name}")
            progress.setValue(index - 1)
            QApplication.processEvents()
            try:
                image = Image.open(path)
                processor.clear_cache()
                result = processor.process(image, settings)
                result.save(Path(output_dir) / f"{path.stem}.png")
                completed += 1
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")

        progress.setValue(len(files))
        message = f"Tamamlanan: {completed}/{len(files)}"
        if errors:
            message += f"\nHatalı: {len(errors)}\n\n" + "\n".join(errors[:8])
        QMessageBox.information(self, "Toplu son işlem", message)
    def _build_model_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self.base_model = QLineEdit(self.config["base_model"])
        self.ip_repo = QLineEdit(self.config["ip_adapter_repo"])
        self.ip_sub = QLineEdit(self.config["ip_adapter_subfolder"])
        self.ip_weight = QLineEdit(self.config["ip_adapter_weight"])
        self.trigger = QLineEdit(self.config.get("lora_trigger", "football sticker caricature"))
        form.addRow("Base model:", self.base_model)
        form.addRow("IP-Adapter repo:", self.ip_repo)
        form.addRow("IP-Adapter alt klasörü:", self.ip_sub)
        form.addRow("IP-Adapter ağırlığı:", self.ip_weight)
        form.addRow("LoRA trigger:", self.trigger)
        save = QPushButton("Model ayarlarını kaydet")
        save.clicked.connect(self.save_config)
        form.addRow(save)
        note = QLabel(
            "Not: EXE yalnızca arayüzü paketler. Diffusion modelleri ilk kullanımda "
            "Hugging Face üzerinden indirilir ve kullanıcı önbelleğinde tutulur."
        )
        note.setWordWrap(True)
        form.addRow(note)
        return w

    def _build_help_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        txt = QPlainTextEdit()
        txt.setReadOnly(True)
        txt.setPlainText(
            "1) Üretim sekmesinde modeli çalıştır.\n"
            "2) Son üretilen görsel otomatik olarak Son İşlem sekmesine aktarılır.\n"
            "3) Son İşlem içinde arka plan, renk, ayrıntı ve kontur ayarlarını "
            "modeli yeniden çalıştırmadan değiştirebilirsin.\n"
            "4) AI — ISNet en kaliteli genel arka plan yöntemidir; ilk kullanımda "
            "model indirebilir.\n"
            "5) Düz beyaz veya tek renk arka planlarda Beyaz/Kenar rengi modları "
            "çok daha hızlıdır.\n"
            "6) Siyah kontur için önce arka planı kaldırmak gerekir.\n"
            "7) Postprocessed PNG Kaydet düğmesi çıktıyı postprocessed klasörüne yazar.\n"
            "8) Klasöre Toplu Uygula aynı ayarları tüm tamamlanmış portrelere uygular.\n"
            "9) EXE'yi güncellemek için build_exe.bat dosyasını yeniden çalıştır."
        )
        v.addWidget(txt)
        return w

    @staticmethod
    def _double(value, minimum, maximum, step):
        w = QDoubleSpinBox()
        w.setRange(minimum, maximum)
        w.setSingleStep(step)
        w.setDecimals(2)
        w.setValue(value)
        return w

    @staticmethod
    def _spin(value, minimum, maximum):
        w = QSpinBox()
        w.setRange(minimum, maximum)
        w.setValue(value)
        return w

    def _path_row(self, edit, fn):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        b = QPushButton("Seç")
        b.clicked.connect(fn)
        h.addWidget(edit)
        h.addWidget(b)
        return w

    def apply_preset(self, preset: str):
        if preset == "identity":
            self.face.setValue(0.86)
            self.caricature.setValue(0.55)
            self.style.setValue(0.75)
            self.colors.setValue(0.18)
            self.guidance.setValue(7.0)
            self.steps.setValue(24)
        elif preset == "style":
            self.face.setValue(0.66)
            self.caricature.setValue(0.78)
            self.style.setValue(1.00)
            self.colors.setValue(0.05)
            self.guidance.setValue(7.5)
            self.steps.setValue(28)
        else:
            self.face.setValue(0.76)
            self.caricature.setValue(0.68)
            self.style.setValue(0.90)
            self.colors.setValue(0.10)
            self.guidance.setValue(7.0)
            self.steps.setValue(26)

    def pick_csv(self):
        p, _ = QFileDialog.getOpenFileName(self, "CSV seç", "", "CSV (*.csv)")
        if p:
            self.csv_edit.setText(p)

    def pick_output(self):
        p = QFileDialog.getExistingDirectory(self, "Çıktı klasörü seç")
        if p:
            self.out_edit.setText(p)

    def pick_lora(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "LoRA seç", "", "SafeTensors (*.safetensors);;Tüm dosyalar (*)"
        )
        if p:
            self.lora_edit.setText(p)

    def save_config(self):
        self.config.update({
            "base_model": self.base_model.text().strip(),
            "ip_adapter_repo": self.ip_repo.text().strip(),
            "ip_adapter_subfolder": self.ip_sub.text().strip(),
            "ip_adapter_weight": self.ip_weight.text().strip(),
            "lora_path": self.lora_edit.text().strip(),
            "lora_trigger": self.trigger.text().strip(),
            "default_prompt": self.prompt.toPlainText().strip(),
            "negative_prompt": self.negative.toPlainText().strip(),
            "output_size": int(self.size.currentText()),
        })
        CONFIG_PATH.write_text(json.dumps(self.config, indent=2, ensure_ascii=False), encoding="utf-8")
        QMessageBox.information(self, "Kaydedildi", "config.json güncellendi.")

    def start_job(self):
        csv_path = Path(self.csv_edit.text().strip())
        if not csv_path.exists():
            QMessageBox.warning(self, "Eksik", "Geçerli bir CSV seç.")
            return
        if self.use_lora.isChecked() and not Path(self.lora_edit.text().strip()).exists():
            QMessageBox.warning(
                self, "LoRA eksik",
                "Stil LoRA açık ama dosya seçilmemiş. LoRA'yı kapat veya dosya seç."
            )
            return

        self.save_config()
        settings = RenderSettings(
            prompt=self.prompt.toPlainText().strip(),
            negative_prompt=self.negative.toPlainText().strip(),
            face_similarity=self.face.value(),
            caricature_strength=self.caricature.value(),
            style_strength=self.style.value(),
            guidance_scale=self.guidance.value(),
            steps=self.steps.value(),
            seed=self.seed.value(),
            output_size=int(self.size.currentText()),
            remove_background=self.remove_bg.isChecked(),
            sharpen=self.sharpen.value(),
            preserve_colors=self.colors.value(),
            use_ip_adapter=self.use_ip.isChecked(),
            use_lora=self.use_lora.isChecked(),
        )

        job = {
            "config": dict(self.config),
            "settings": asdict(settings),
            "csv_path": str(csv_path),
            "output_dir": self.out_edit.text().strip(),
            "id_col": self.id_col.text().strip(),
            "name_col": self.name_col.text().strip(),
            "url_col": self.url_col.text().strip(),
            "start_index": self.start_index.value(),
            "limit": self.limit.value(),
            "skip_existing": self.skip_existing.isChecked(),
            "random_seed": self.random_seed.isChecked(),
        }

        self.thread = QThread()
        self.worker = Worker(job)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.preview.connect(self.on_preview)
        self.worker.log.connect(self.append_log)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setValue(0)
        self.thread.start()

    def stop_job(self):
        if self.worker:
            self.worker.cancel()
            self.status.setText("Durdurma isteği gönderildi...")

    def on_progress(self, current, total, text):
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(current)
        self.status.setText(f"{current}/{total} — {text}")

    def on_preview(self, source, result, name, output_path):
        self.source_preview.setPixmap(pil_to_pixmap(source))
        self.result_preview.setPixmap(pil_to_pixmap(result))
        self.status.setText(name)
        self._set_post_source(result, name, output_path)

    def append_log(self, text):
        self.log.appendPlainText(text)

    def on_finished(self, stats):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.append_log(f"Tamamlandı: {stats}")
        QMessageBox.information(self, "Bitti", json.dumps(stats, ensure_ascii=False))

    def on_failed(self, trace):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.append_log(trace)
        QMessageBox.critical(self, "Kritik hata", trace[-3000:])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
