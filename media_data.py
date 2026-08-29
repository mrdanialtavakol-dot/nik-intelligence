from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ASSET_DIR = BASE_DIR / "assets"


@dataclass(frozen=True)
class MediaVideo:
    media_id: str
    title: str
    filename: str
    duration: float
    format: str
    topic: str
    hook: str
    proof: str
    cta: str
    seed: int

    @property
    def path(self) -> Path:
        return ASSET_DIR / "videos" / self.filename


@dataclass(frozen=True)
class MediaImage:
    media_id: str
    title: str
    filename: str
    category: str
    visual_note: str

    @property
    def path(self) -> Path:
        return ASSET_DIR / "images" / self.filename


MEDIA_VIDEOS: List[MediaVideo] = [
    MediaVideo(
        "V01",
        "کاربرد نیک‌پوز و باشگاه مشتریان",
        "video-01.mp4",
        43.05,
        "Reel / Vertical",
        "Retention / Customer Club",
        "کاربرد نیک‌پوز",
        "نمایش پنل و ثبت مشتری",
        "خرید و تست دستگاه",
        101,
    ),
    MediaVideo(
        "V02",
        "تراکت، کارت ویزیت و مسیر ارتباطی",
        "video-02.mp4",
        40.87,
        "Story / Vertical",
        "Traditional vs Smart Tools",
        "شروع در فضای واقعی کسب‌وکار",
        "نمایش نقشه، موبایل و دستگاه",
        "هدایت به لینک",
        202,
    ),
    MediaVideo(
        "V03",
        "منفعت استفاده از دستگاه",
        "video-03.mp4",
        57.94,
        "Story / Vertical",
        "Customer Retention",
        "چند بار شده مشتری را از دست بدهی؟",
        "توضیح ارتباط دوباره با مشتری",
        "اقدام برای تهیه دستگاه",
        303,
    ),
    MediaVideo(
        "V04",
        "ارسال منطقه‌ای و جذب مشتری",
        "video-04.mp4",
        105.58,
        "Long Reel / Vertical",
        "Regional SMS",
        "از پول تبلیغات چقدر استفاده می‌کنی؟",
        "نقشه ارسال منطقه‌ای و نمونه پنل",
        "تبدیل جذب به دیتابیس مشتری",
        404,
    ),
    MediaVideo(
        "V05",
        "بازار شماره مشتری و پیشنهاد خرید",
        "video-05.mp4",
        43.69,
        "Reel / Vertical",
        "Customer Data Value",
        "ارزش بازار شماره مشتری",
        "SMS، پنل و نمایش محصول",
        "کامنت / لینک خرید",
        505,
    ),
]

MEDIA_IMAGES: List[MediaImage] = [
    MediaImage("I01", "تو شبیه کدومی؟", "cover-01.webp", "Reel Cover", "دو کاراکتر + تضاد انتخاب"),
    MediaImage("I02", "مشتری VIP برای مغازه خودت", "cover-02.webp", "Reel Cover", "تمرکز روی محصول و تفاوت دستگاه"),
    MediaImage("I03", "آچار فرانسه برای کسب‌وکار", "cover-03.webp", "Reel Cover", "Metaphor / ابزار همه‌کاره"),
    MediaImage("I04", "کمپین زمانی", "cover-04.webp", "Story Cover", "Urgency / بازه تاریخی"),
    MediaImage("I05", "وابستگی خطرناک", "cover-05.webp", "Reel Cover", "بحران اینترنت / Pain Point"),
    MediaImage("I06", "تبلیغ با بودجه کم", "cover-06.webp", "Reel Cover", "Hook اقتصادی / شوک قیمتی"),
]


def get_video(media_id: str) -> MediaVideo:
    for item in MEDIA_VIDEOS:
        if item.media_id == media_id:
            return item
    return MEDIA_VIDEOS[0]


def synthetic_video_timeline(video: MediaVideo) -> pd.DataFrame:
    """Deterministic demo timeline. It is NOT Instagram event-level data."""
    rng = np.random.default_rng(video.seed)
    duration = max(10, int(round(video.duration)))
    t = np.arange(0, duration + 1, dtype=float)

    # Smooth decay with deterministic content moments. The shape is intentionally plausible,
    # but is labelled synthetic everywhere in the UI.
    base_decay = 0.82 * np.exp(-t / (duration * 0.64)) + 0.18
    hook_bump = 0.055 * np.exp(-0.5 * ((t - min(3.0, duration * 0.08)) / 1.35) ** 2)
    proof_bump = 0.045 * np.exp(-0.5 * ((t - duration * 0.48) / max(2.2, duration * 0.045)) ** 2)
    cta_bump = 0.035 * np.exp(-0.5 * ((t - duration * 0.84) / max(2.0, duration * 0.04)) ** 2)
    noise = rng.normal(0, 0.006, size=len(t))
    retention = np.clip(base_decay + hook_bump + proof_bump + cta_bump + noise, 0.18, 1.0)
    retention[0] = 1.0
    # Enforce mostly decreasing behavior while preserving small rewatch bumps.
    for i in range(1, len(retention)):
        retention[i] = min(retention[i], retention[i - 1] + 0.018)

    interaction = (
        0.15
        + 1.35 * np.exp(-0.5 * ((t - duration * 0.50) / max(2.0, duration * 0.05)) ** 2)
        + 1.85 * np.exp(-0.5 * ((t - duration * 0.84) / max(1.7, duration * 0.035)) ** 2)
        + rng.uniform(0.0, 0.18, size=len(t))
    )
    click_signal = (
        0.03
        + 0.95 * np.exp(-0.5 * ((t - duration * 0.86) / max(1.6, duration * 0.03)) ** 2)
        + 0.18 * np.exp(-0.5 * ((t - duration * 0.52) / max(2.0, duration * 0.045)) ** 2)
    )

    return pd.DataFrame(
        {
            "second": t.astype(int),
            "retention": retention,
            "interaction_signal": interaction,
            "click_signal": click_signal,
        }
    )


def synthetic_video_events(video: MediaVideo) -> pd.DataFrame:
    d = video.duration
    rows = [
        (max(1, round(d * 0.04)), "Hook", video.hook, "شروع توجه"),
        (max(3, round(d * 0.22)), "Drop", "نقطه افت آزمایشی", "افت احتمالی"),
        (max(5, round(d * 0.48)), "Proof", video.proof, "اوج تعامل"),
        (max(7, round(d * 0.70)), "Rewatch", "بازگشت توجه / Replay Proxy", "تعامل"),
        (max(9, round(d * 0.86)), "CTA", video.cta, "Click Proxy"),
    ]
    return pd.DataFrame(rows, columns=["second", "event", "description", "signal"])


def synthetic_video_metrics(video: MediaVideo) -> dict[str, float]:
    df = synthetic_video_timeline(video)
    duration = max(video.duration, 1.0)
    idx3 = min(3, len(df) - 1)
    hook_hold = float(df.iloc[idx3]["retention"])
    completion = float(df.iloc[-1]["retention"])
    # Area under retention curve approximates average watch time for the demo.
    avg_watch = float(np.trapezoid(df["retention"].to_numpy(), df["second"].to_numpy()))
    cta_peak = float(df["click_signal"].max())
    interaction_peak_second = float(df.loc[df["interaction_signal"].idxmax(), "second"])
    return {
        "hook_hold": hook_hold,
        "completion_rate": completion,
        "avg_watch_time": avg_watch,
        "cta_click_signal": cta_peak,
        "interaction_peak_second": interaction_peak_second,
    }
