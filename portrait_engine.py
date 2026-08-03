
from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass
class RenderSettings:
    prompt: str
    negative_prompt: str
    face_similarity: float = 0.82
    caricature_strength: float = 0.55
    style_strength: float = 0.85
    guidance_scale: float = 7.0
    steps: int = 24
    seed: int = 12345
    output_size: int = 512
    remove_background: bool = False
    sharpen: float = 1.15
    preserve_colors: float = 0.65
    use_ip_adapter: bool = True
    use_lora: bool = True


class PortraitEngine:
    def __init__(
        self,
        base_model: str,
        ip_adapter_repo: str,
        ip_adapter_subfolder: str,
        ip_adapter_weight: str,
        lora_path: str = "",
        lora_trigger: str = "",
        status_cb: Optional[Callable[[str], None]] = None,
    ):
        self.base_model = base_model
        self.ip_adapter_repo = ip_adapter_repo
        self.ip_adapter_subfolder = ip_adapter_subfolder
        self.ip_adapter_weight = ip_adapter_weight
        self.lora_path = lora_path.strip()
        self.lora_trigger = lora_trigger.strip()
        self.status_cb = status_cb or (lambda _: None)
        self.pipe = None
        self.loaded_lora_path = None

    def _status(self, text: str) -> None:
        self.status_cb(text)

    @staticmethod
    def _device_dtype():
        if torch.cuda.is_available():
            return "cuda", torch.float16
        return "cpu", torch.float32

    def load(self, settings: RenderSettings) -> None:
        if self.pipe is not None:
            return

        from diffusers import (
            DPMSolverMultistepScheduler,
            StableDiffusionImg2ImgPipeline,
        )

        device, dtype = self._device_dtype()
        self._status(f"Model yükleniyor: {self.base_model}")

        # Pipeline'ı önce yerel değişkende kuruyoruz. Bir adapter yükleme hatası olursa
        # yarım kalmış/bzulmuş nesne sonraki oyuncularda yeniden kullanılmaz.
        pipe = None
        try:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.base_model,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                use_karras_sigmas=True,
            )

            # ÖNEMLİ:
            # PyTorch 2.x zaten SDPA kullanır. IP-Adapter yüklenmeden önce
            # enable_attention_slicing() çağırmak SlicedAttnProcessor çakışmasına
            # yol açabiliyor. Bu nedenle attention slicing tamamen kaldırıldı.
            #
            # Adapter'ları model CPU offload'dan ÖNCE yüklüyoruz.
            if settings.use_ip_adapter:
                self._load_ip_adapter(pipe)

            if settings.use_lora and self.lora_path:
                self._load_lora(pipe, self.lora_path)

            # VAE slicing güvenlidir ve özellikle düşük VRAM'de faydalıdır.
            pipe.enable_vae_slicing()

            if device == "cuda":
                # 4 GB sınıfı kartlarda model parçalarını gerektiğinde GPU'ya taşır.
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cpu")

            self.pipe = pipe
            self._status("Model hazır.")

        except Exception:
            # İlk oyuncudaki model yükleme hatasının kalan oyunculara
            # NoneType/bozuk pipeline olarak taşınmasını engelle.
            self.pipe = None
            if pipe is not None:
                try:
                    del pipe
                except Exception:
                    pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    def _load_ip_adapter(self, pipe) -> None:
        try:
            self._status("IP-Adapter yükleniyor...")
            pipe.load_ip_adapter(
                self.ip_adapter_repo,
                subfolder=self.ip_adapter_subfolder,
                weight_name=self.ip_adapter_weight,
            )
        except Exception as exc:
            raise RuntimeError(
                "IP-Adapter yüklenemedi. Model adı/dosyası veya Diffusers attention "
                "yapılandırması uyumsuz olabilir.\n"
                f"Repo: {self.ip_adapter_repo}\n"
                f"Alt klasör: {self.ip_adapter_subfolder}\n"
                f"Ağırlık: {self.ip_adapter_weight}\n"
                f"Ayrıntı: {exc}"
            ) from exc

    def _load_lora(self, pipe, path: str) -> None:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"LoRA bulunamadı: {path}")

        self._status(f"Stil LoRA yükleniyor: {path_obj.name}")
        if path_obj.is_file():
            pipe.load_lora_weights(
                str(path_obj.parent),
                weight_name=path_obj.name,
                adapter_name="portrait_style",
            )
        else:
            pipe.load_lora_weights(
                str(path_obj),
                adapter_name="portrait_style",
            )
        self.loaded_lora_path = str(path_obj.resolve())

    @staticmethod
    def prepare_source(image: Image.Image, size: int) -> Image.Image:
        image = ImageOps.exif_transpose(image).convert("RGB")

        # Orijinal vesikalık duruşunu korur; kare tuvale taşırken kırpma yerine padding kullanır.
        w, h = image.size
        margin = int(max(w, h) * 0.08)
        canvas_side = max(w, h) + margin * 2
        fill = tuple(np.asarray(image.resize((1, 1))).reshape(-1).astype(int).tolist())
        canvas = Image.new("RGB", (canvas_side, canvas_side), fill)
        canvas.paste(image, ((canvas_side - w) // 2, (canvas_side - h) // 2))
        canvas = canvas.resize((size, size), Image.Resampling.LANCZOS)
        return canvas

    @staticmethod
    def _blend_colors(result: Image.Image, source: Image.Image, amount: float) -> Image.Image:
        amount = max(0.0, min(1.0, amount))
        if amount <= 0:
            return result
        source = source.resize(result.size, Image.Resampling.LANCZOS).convert("RGB")
        # Kaynağın düşük frekanslı renk bilgisini geri verir; yüz detayını kopyalamaz.
        color_layer = source.filter(ImageFilter.GaussianBlur(radius=max(2, result.width // 80)))
        return Image.blend(result.convert("RGB"), color_layer, amount * 0.28)

    @staticmethod
    def _remove_background(image: Image.Image) -> Image.Image:
        from rembg import remove
        return remove(image.convert("RGBA"))

    def render(self, source: Image.Image, settings: RenderSettings) -> Image.Image:
        self.load(settings)

        size = int(settings.output_size)
        source_prepared = self.prepare_source(source, size)

        face = max(0.0, min(1.0, settings.face_similarity))
        caricature = max(0.05, min(0.95, settings.caricature_strength))
        style = max(0.0, min(1.5, settings.style_strength))

        # v1.2: Kimlik koruma ile stil dönüşümü birbirinden ayrıldı.
        # Yüz/kimlik gücü yalnızca IP-Adapter scale değerini kontrol eder.
        # Karikatür/denoise değeri img2img strength'i doğrudan kontrol eder.
        effective_strength = max(0.10, min(0.90, caricature))

        prompt = settings.prompt.strip()
        if settings.use_lora and self.lora_trigger:
            prompt = f"{self.lora_trigger}, {prompt}"

        if settings.use_ip_adapter:
            self.pipe.set_ip_adapter_scale(face)

        if settings.use_lora and self.loaded_lora_path:
            self.pipe.set_adapters(["portrait_style"], adapter_weights=[style])

        generator_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(settings.seed))

        kwargs = dict(
            prompt=prompt,
            negative_prompt=settings.negative_prompt,
            image=source_prepared,
            strength=effective_strength,
            guidance_scale=float(settings.guidance_scale),
            num_inference_steps=int(settings.steps),
            generator=generator,
        )
        if settings.use_ip_adapter:
            kwargs["ip_adapter_image"] = source_prepared

        result = self.pipe(**kwargs).images[0].convert("RGB")
        result = self._blend_colors(result, source_prepared, settings.preserve_colors)

        if settings.sharpen != 1.0:
            result = ImageEnhance.Sharpness(result).enhance(float(settings.sharpen))

        if settings.remove_background:
            result = self._remove_background(result)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result
