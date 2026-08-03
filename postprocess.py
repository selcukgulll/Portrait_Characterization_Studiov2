
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass(frozen=True)
class PostProcessSettings:
    background_mode: str = "off"  # off, isnet, u2net, border, white
    background_tolerance: int = 28
    feather: int = 2
    alpha_expand: int = 0
    dehalo: int = 1
    keep_largest: bool = True
    fill_holes: bool = True

    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    sharpness: float = 1.0
    detail: float = 0.0
    hue: int = 0
    gamma: float = 1.0

    outline_enabled: bool = False
    outline_thickness: int = 4
    outline_opacity: float = 1.0
    outline_softness: int = 0
    internal_line_strength: float = 0.0


_REMBG_SESSIONS: dict[str, object] = {}


def _image_key(image: Image.Image) -> str:
    rgba = ImageOps.exif_transpose(image).convert("RGBA")
    h = sha1()
    h.update(str(rgba.size).encode("ascii"))
    h.update(rgba.tobytes())
    return h.hexdigest()


def _get_rembg_session(model_name: str):
    if model_name in _REMBG_SESSIONS:
        return _REMBG_SESSIONS[model_name]

    from rembg import new_session

    try:
        session = new_session(model_name)
    except Exception:
        # Model adı mevcut rembg sürümünde desteklenmiyorsa varsayılan modele dön.
        session = new_session()
    _REMBG_SESSIONS[model_name] = session
    return session


def _estimate_border_color(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    band = max(2, min(h, w) // 32)
    border = np.concatenate(
        [
            rgb[:band, :, :].reshape(-1, 3),
            rgb[-band:, :, :].reshape(-1, 3),
            rgb[:, :band, :].reshape(-1, 3),
            rgb[:, -band:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(border, axis=0).astype(np.uint8)


def _border_connected_background(rgb: np.ndarray, tolerance: int, fixed_white: bool) -> np.ndarray:
    if fixed_white:
        bg_rgb = np.array([255, 255, 255], dtype=np.uint8)
    else:
        bg_rgb = _estimate_border_color(rgb)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(bg_rgb.reshape(1, 1, 3), cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    distance = np.linalg.norm(lab - bg_lab, axis=2)

    # Kullanıcı 0-100 aralığı görür. LAB mesafesine kontrollü biçimde ölçeklenir.
    threshold = 4.0 + float(tolerance) * 1.55
    candidate = (distance <= threshold).astype(np.uint8)

    count, labels = cv2.connectedComponents(candidate, connectivity=8)
    touching: set[int] = set()
    touching.update(np.unique(labels[0, :]).tolist())
    touching.update(np.unique(labels[-1, :]).tolist())
    touching.update(np.unique(labels[:, 0]).tolist())
    touching.update(np.unique(labels[:, -1]).tolist())
    touching.discard(0)

    if not touching:
        background = candidate.astype(bool)
    else:
        background = np.isin(labels, list(touching))

    alpha = np.where(background, 0, 255).astype(np.uint8)
    return alpha


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    binary = np.where(mask > 20, 255, 0).astype(np.uint8)
    flood = binary.copy()
    h, w = binary.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)

    # Birkaç köşeyi dene; konu köşeye değse bile başka bir arka plan başlangıcı bulunabilir.
    seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    for seed in seeds:
        if flood[seed[1], seed[0]] == 0:
            cv2.floodFill(flood, flood_mask, seed, 255)

    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(binary, holes)


def _keep_largest(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 20).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return mask

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    keep = labels == largest

    # Orijinal yumuşak alfa değerlerini sadece ana bileşende koru.
    result = np.zeros_like(mask)
    result[keep] = mask[keep]
    return result


def _cleanup_alpha(alpha: np.ndarray, settings: PostProcessSettings) -> np.ndarray:
    alpha = alpha.astype(np.uint8)

    if settings.fill_holes:
        filled = _fill_holes(alpha)
        alpha = np.maximum(alpha, filled)

    if settings.keep_largest:
        alpha = _keep_largest(alpha)

    amount = int(settings.alpha_expand)
    if amount != 0:
        kernel_size = max(3, abs(amount) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        if amount > 0:
            alpha = cv2.dilate(alpha, kernel, iterations=1)
        else:
            alpha = cv2.erode(alpha, kernel, iterations=1)

    # Dehalo, maskeyi çok hafif içeri alarak eski arka plan saçaklarını keser.
    if settings.dehalo > 0:
        size = max(3, int(settings.dehalo) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        alpha = cv2.erode(alpha, kernel, iterations=1)

    if settings.feather > 0:
        radius = int(settings.feather)
        alpha = cv2.GaussianBlur(alpha, (radius * 2 + 1, radius * 2 + 1), 0)

    return alpha


def _decontaminate_edges(
    rgb: np.ndarray,
    alpha: np.ndarray,
    background_color: np.ndarray,
    strength: int,
) -> np.ndarray:
    if strength <= 0:
        return rgb

    a = alpha.astype(np.float32) / 255.0
    safe = np.maximum(a, 0.08)[..., None]
    bg = background_color.astype(np.float32).reshape(1, 1, 3)
    recovered = (rgb.astype(np.float32) - bg * (1.0 - a[..., None])) / safe
    recovered = np.clip(recovered, 0, 255)

    edge = ((a > 0.02) & (a < 0.98)).astype(np.float32)[..., None]
    blend = min(1.0, strength / 4.0) * edge
    return np.clip(rgb * (1.0 - blend) + recovered * blend, 0, 255).astype(np.uint8)


def _remove_background(image: Image.Image, settings: PostProcessSettings) -> Image.Image:
    source = ImageOps.exif_transpose(image).convert("RGBA")
    rgb = np.asarray(source.convert("RGB"))
    mode = settings.background_mode

    if mode in {"isnet", "u2net"}:
        from rembg import remove

        model_name = "isnet-general-use" if mode == "isnet" else "u2net"
        session = _get_rembg_session(model_name)

        kwargs = {
            "session": session,
            "alpha_matting": True,
            "alpha_matting_foreground_threshold": 235,
            "alpha_matting_background_threshold": 15,
            "alpha_matting_erode_size": 7,
            "post_process_mask": True,
        }
        try:
            result = remove(source, **kwargs).convert("RGBA")
        except TypeError:
            # Eski rembg sürümleri bazı kwargs'ları kabul etmeyebilir.
            result = remove(source, session=session).convert("RGBA")
        rgba = np.asarray(result).copy()
        alpha = rgba[:, :, 3]
        base_rgb = rgba[:, :, :3]
    elif mode in {"border", "white"}:
        alpha = _border_connected_background(
            rgb,
            tolerance=int(settings.background_tolerance),
            fixed_white=(mode == "white"),
        )
        base_rgb = rgb.copy()
    else:
        return source

    alpha = _cleanup_alpha(alpha, settings)
    bg_color = _estimate_border_color(rgb)
    base_rgb = _decontaminate_edges(base_rgb, alpha, bg_color, settings.dehalo)

    rgba = np.dstack([base_rgb, alpha]).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def _shift_hue(rgb: Image.Image, degrees: int) -> Image.Image:
    if degrees == 0:
        return rgb
    hsv = np.asarray(rgb.convert("HSV")).copy()
    shift = int(round(degrees / 360.0 * 255.0))
    hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + shift) % 256
    return Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")


def _apply_gamma(rgb: Image.Image, gamma: float) -> Image.Image:
    gamma = max(0.05, float(gamma))
    if abs(gamma - 1.0) < 1e-4:
        return rgb
    # gamma > 1 görüntüyü açar; kullanıcı açısından daha sezgisel davranır.
    inv = 1.0 / gamma
    lut = np.array([round((i / 255.0) ** inv * 255.0) for i in range(256)], dtype=np.uint8)
    arr = lut[np.asarray(rgb.convert("RGB"))]
    return Image.fromarray(arr, "RGB")


def _apply_color_adjustments(image: Image.Image, settings: PostProcessSettings) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    rgb = rgba.convert("RGB")

    rgb = _apply_gamma(rgb, settings.gamma)
    rgb = _shift_hue(rgb, int(settings.hue))
    rgb = ImageEnhance.Brightness(rgb).enhance(float(settings.brightness))
    rgb = ImageEnhance.Contrast(rgb).enhance(float(settings.contrast))
    rgb = ImageEnhance.Color(rgb).enhance(float(settings.saturation))

    detail_amount = max(0.0, float(settings.detail))
    while detail_amount > 0:
        step = min(1.0, detail_amount)
        detailed = rgb.filter(ImageFilter.DETAIL)
        rgb = Image.blend(rgb, detailed, step)
        detail_amount -= 1.0

    rgb = ImageEnhance.Sharpness(rgb).enhance(float(settings.sharpness))
    rgb.putalpha(alpha)
    return rgb


def _add_internal_lines(image: Image.Image, strength: float) -> Image.Image:
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0:
        return image

    rgba = np.asarray(image.convert("RGBA")).copy()
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    median = float(np.median(gray[alpha > 20])) if np.any(alpha > 20) else 128.0
    low = int(max(20, 0.55 * median))
    high = int(min(240, 1.35 * median))
    edges = cv2.Canny(gray, low, high)
    edges = cv2.bitwise_and(edges, np.where(alpha > 20, 255, 0).astype(np.uint8))

    edge_alpha = (edges.astype(np.float32) / 255.0 * strength * 170.0).astype(np.uint8)
    dark = np.zeros_like(rgb)
    mix = edge_alpha.astype(np.float32)[..., None] / 255.0
    rgba[:, :, :3] = np.clip(rgb * (1.0 - mix) + dark * mix, 0, 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def _add_outline(image: Image.Image, settings: PostProcessSettings) -> Image.Image:
    if not settings.outline_enabled or settings.outline_thickness <= 0:
        return image

    subject = image.convert("RGBA")
    alpha = np.asarray(subject.getchannel("A"))
    thickness = max(1, int(settings.outline_thickness))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (thickness * 2 + 1, thickness * 2 + 1),
    )
    dilated = cv2.dilate(alpha, kernel, iterations=1)
    outline_alpha = np.clip(dilated.astype(np.int16) - alpha.astype(np.int16), 0, 255).astype(np.uint8)

    if settings.outline_softness > 0:
        radius = int(settings.outline_softness)
        outline_alpha = cv2.GaussianBlur(
            outline_alpha,
            (radius * 2 + 1, radius * 2 + 1),
            0,
        )

    outline_alpha = np.clip(
        outline_alpha.astype(np.float32) * float(settings.outline_opacity),
        0,
        255,
    ).astype(np.uint8)

    outline = Image.new("RGBA", subject.size, (0, 0, 0, 0))
    outline.putalpha(Image.fromarray(outline_alpha, "L"))
    return Image.alpha_composite(outline, subject)


class PostProcessor:
    """Tek görsel için arka plan maskesini cache'leyerek hızlı canlı önizleme sağlar."""

    def __init__(self):
        self._source_key: Optional[str] = None
        self._background_key: Optional[tuple] = None
        self._background_result: Optional[Image.Image] = None

    def clear_cache(self) -> None:
        self._source_key = None
        self._background_key = None
        self._background_result = None

    def process(self, image: Image.Image, settings: PostProcessSettings) -> Image.Image:
        source = ImageOps.exif_transpose(image).convert("RGBA")
        source_key = _image_key(source)
        background_key = (
            settings.background_mode,
            int(settings.background_tolerance),
            int(settings.feather),
            int(settings.alpha_expand),
            int(settings.dehalo),
            bool(settings.keep_largest),
            bool(settings.fill_holes),
        )

        if settings.background_mode == "off":
            base = source
        elif (
            self._source_key == source_key
            and self._background_key == background_key
            and self._background_result is not None
        ):
            base = self._background_result.copy()
        else:
            base = _remove_background(source, settings)
            self._source_key = source_key
            self._background_key = background_key
            self._background_result = base.copy()

        result = _apply_color_adjustments(base, settings)
        result = _add_internal_lines(result, settings.internal_line_strength)
        result = _add_outline(result, settings)
        return result
