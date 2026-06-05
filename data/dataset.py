
from __future__ import annotations

import csv
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import fastf1
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Telemetry column aliases (fastf1 may vary by version)
# ---------------------------------------------------------------------------
_SPEED_COL = "Speed"
_THROTTLE_COL = "Throttle"
_BRAKE_COL = "Brake"
_GEAR_COL = "nGear"
_DRS_COL = "DRS"
_STEER_COL = "nSteeringAngle"
_TIME_COL = "Time"


# ---------------------------------------------------------------------------
# 1. TelemetryPromptGenerator
# ---------------------------------------------------------------------------

class TelemetryPromptGenerator:
    """
    Heuristic rule-based converter from scalar telemetry → natural language.

    All input values are expected in the range that fastf1 returns them:
      - speed    : km/h
      - throttle : 0–100 (percentage)
      - brake    : 0–100 (percentage)
      - steering : degrees (will be normalised internally to [-1, 1])
      - gear     : integer 1–8 (optional)
      - drs      : integer 0/8/10/14 (optional; >0 means open)

    Example output
    --------------
    "The driver brakes aggressively and turns sharply right at high speed,
     in gear 3."
    """

    # -- Speed bands (km/h) -------------------------------------------------
    _SPEED_BANDS: List[Tuple[float, str]] = [
        (100,  "very low speed"),
        (180,  "low speed"),
        (240,  "medium speed"),
        (300,  "high speed"),
        (float("inf"), "very high speed"),
    ]

    # -- Normalised thresholds (0–1) ----------------------------------------
    _THROTTLE_OFF   = 0.05
    _THROTTLE_PART  = 0.50
    _THROTTLE_HEAVY = 0.85

    _BRAKE_NONE     = 0.03
    _BRAKE_LIGHT    = 0.25
    _BRAKE_MOD      = 0.60

    # Steering: max lock usually ~450°; normalise to ±1 using ±250° range
    _STEER_MAX_DEG    = 250.0
    _STEER_STRAIGHT   = 0.04
    _STEER_SLIGHT     = 0.18
    _STEER_MODERATE   = 0.50

    def _speed_desc(self, speed_kmh: float) -> str:
        for threshold, label in self._SPEED_BANDS:
            if speed_kmh < threshold:
                return label
        return "very high speed"

    def _throttle_clause(self, throttle_pct: float) -> Optional[str]:
        t = throttle_pct / 100.0
        if t < self._THROTTLE_OFF:
            return None
        if t < self._THROTTLE_PART:
            return "partially applies throttle"
        if t < self._THROTTLE_HEAVY:
            return "applies heavy throttle"
        return "is flat out on the throttle"

    def _brake_clause(self, brake_pct: float) -> Optional[str]:
        b = brake_pct / 100.0
        if b < self._BRAKE_NONE:
            return None
        if b < self._BRAKE_LIGHT:
            return "brakes lightly"
        if b < self._BRAKE_MOD:
            return "brakes moderately"
        return "brakes aggressively"

    def _steer_clause(self, steer_deg: float) -> Optional[str]:
        norm = max(-1.0, min(1.0, steer_deg / self._STEER_MAX_DEG))
        mag = abs(norm)
        direction = "right" if norm > 0 else "left"
        if mag < self._STEER_STRAIGHT:
            return None
        if mag < self._STEER_SLIGHT:
            return f"steers slightly {direction}"
        if mag < self._STEER_MODERATE:
            return f"turns {direction}"
        return f"turns sharply {direction}"

    # ------------------------------------------------------------------

    def generate(
        self,
        speed: float,
        throttle: float,
        brake: float,
        steering: float,
        gear: Optional[int] = None,
        drs: Optional[int] = None,
    ) -> str:
        """
        Build a single natural-language prompt from one telemetry sample.

        Parameters
        ----------
        speed    : float – km/h
        throttle : float – 0-100
        brake    : float – 0-100
        steering : float – degrees (fastf1 nSteeringAngle)
        gear     : int   – optional current gear
        drs      : int   – optional DRS value (>0 = open)
        """
        clauses: List[str] = []

        brake_txt    = self._brake_clause(brake)
        throttle_txt = self._throttle_clause(throttle)
        steer_txt    = self._steer_clause(steering)

        # Brake takes priority over throttle
        if brake_txt:
            clauses.append(brake_txt)
        elif throttle_txt:
            clauses.append(throttle_txt)

        if steer_txt:
            clauses.append(steer_txt)

        speed_desc = self._speed_desc(speed)

        if not clauses:
            base = f"The driver coasts at {speed_desc}"
        elif len(clauses) == 1:
            base = f"The driver {clauses[0]} at {speed_desc}"
        else:
            joined = ", ".join(clauses[:-1]) + f" and {clauses[-1]}"
            base = f"The driver {joined} at {speed_desc}"

        suffix_parts: List[str] = []
        if gear is not None and gear > 0:
            suffix_parts.append(f"in gear {gear}")
        if drs is not None and drs > 0:
            suffix_parts.append("DRS is open")

        if suffix_parts:
            prompt = base + ", " + ", ".join(suffix_parts) + "."
        else:
            prompt = base + "."

        return prompt

    def generate_dataframe(self, df: pd.DataFrame) -> List[str]:
        """Vectorised generation over a telemetry DataFrame."""
        prompts = []
        for _, row in df.iterrows():
            prompts.append(
                self.generate(
                    speed=float(row.get(_SPEED_COL, 0)),
                    throttle=float(row.get(_THROTTLE_COL, 0)),
                    brake=float(row.get(_BRAKE_COL, 0)),
                    steering=float(row.get(_STEER_COL, 0)),
                    gear=int(row[_GEAR_COL]) if _GEAR_COL in row else None,
                    drs=int(row[_DRS_COL]) if _DRS_COL in row else None,
                )
            )
        return prompts


# ---------------------------------------------------------------------------
# 2. FastF1TelemetryLoader
# ---------------------------------------------------------------------------

@dataclass
class TelemetrySample:
    timestamp_ms: float
    speed: float
    throttle: float
    brake: float
    steering: float
    gear: int
    drs: int
    prompt: str


class FastF1TelemetryLoader:
    """
    Downloads / caches telemetry for a specific session + driver
    and exposes a method to retrieve the sample nearest to a
    given video timestamp.

    Parameters
    ----------
    year        : int  – e.g. 2023
    gp          : str  – e.g. 'Monaco'
    session_type: str  – 'R' | 'Q' | 'FP1' | 'FP2' | 'FP3'
    driver      : str  – three-letter code, e.g. 'VER'
    lap_number  : int  – 1-indexed; -1 = fastest lap
    cache_dir   : str  – directory for fastf1's local cache
    """

    def __init__(
        self,
        year: int,
        gp: str,
        session_type: str,
        driver: str,
        lap_number: int = -1,
        cache_dir: str = "./data/f1_cache",
    ) -> None:
        self.year = year
        self.gp = gp
        self.session_type = session_type
        self.driver = driver
        self.lap_number = lap_number
        self.cache_dir = cache_dir

        self._prompt_gen = TelemetryPromptGenerator()
        self._samples: Optional[List[TelemetrySample]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> List[TelemetrySample]:
        """Load telemetry and return list of TelemetrySample."""
        if self._samples is not None:
            return self._samples

        os.makedirs(self.cache_dir, exist_ok=True)
        fastf1.Cache.enable_cache(self.cache_dir)

        logger.info(
            "Loading FastF1 session %d %s %s driver=%s",
            self.year, self.gp, self.session_type, self.driver,
        )
        session = fastf1.get_session(self.year, self.gp, self.session_type)
        session.load(telemetry=True, weather=False, messages=False)

        laps = session.laps.pick_drivers(self.driver)
        if len(laps) == 0:
            raise ValueError(
                f"No laps found for driver {self.driver} in session."
            )

        if self.lap_number == -1:
            lap = laps.pick_fastest()
        else:
            lap = laps[laps["LapNumber"] == self.lap_number].iloc[0]

        telemetry: pd.DataFrame = lap.get_telemetry()
        telemetry = telemetry.reset_index(drop=True)

        # Normalise steering to degrees if stored as raw channel
        if _STEER_COL not in telemetry.columns:
            # Some fastf1 versions expose it differently
            steer_col = "SteeringAngle" if "SteeringAngle" in telemetry.columns else None
            if steer_col:
                telemetry[_STEER_COL] = telemetry[steer_col]
            else:
                telemetry[_STEER_COL] = 0.0

        prompts = self._prompt_gen.generate_dataframe(telemetry)

        self._samples = []
        for i, row in telemetry.iterrows():
            ts = row[_TIME_COL].total_seconds() * 1000  # → ms
            self._samples.append(
                TelemetrySample(
                    timestamp_ms=float(ts),
                    speed=float(row.get(_SPEED_COL, 0)),
                    throttle=float(row.get(_THROTTLE_COL, 0)),
                    brake=float(row.get(_BRAKE_COL, 0)),
                    steering=float(row.get(_STEER_COL, 0)),
                    gear=int(row.get(_GEAR_COL, 1)),
                    drs=int(row.get(_DRS_COL, 0)),
                    prompt=prompts[i],
                )
            )

        logger.info("Loaded %d telemetry samples.", len(self._samples))
        return self._samples

    def get_nearest(self, timestamp_ms: float) -> TelemetrySample:
        """Return the telemetry sample closest to timestamp_ms."""
        samples = self.load()
        idx = min(
            range(len(samples)),
            key=lambda i: abs(samples[i].timestamp_ms - timestamp_ms),
        )
        return samples[idx]

    def save_csv(self, path: str) -> None:
        """Persist loaded samples to a CSV for offline use."""
        samples = self.load()
        rows = [
            {
                "timestamp_ms": s.timestamp_ms,
                "speed": s.speed,
                "throttle": s.throttle,
                "brake": s.brake,
                "steering": s.steering,
                "gear": s.gear,
                "drs": s.drs,
                "prompt": s.prompt,
            }
            for s in samples
        ]
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        logger.info("Saved telemetry CSV → %s", path)

    @classmethod
    def from_csv(cls, path: str) -> "FastF1TelemetryLoader":
        """
        Load a pre-saved telemetry CSV (bypasses fastf1 entirely).
        Returns a loader whose _samples are already populated.
        """
        loader = cls.__new__(cls)
        loader._prompt_gen = TelemetryPromptGenerator()
        df = pd.read_csv(path)
        loader._samples = [
            TelemetrySample(
                timestamp_ms=float(row["timestamp_ms"]),
                speed=float(row["speed"]),
                throttle=float(row["throttle"]),
                brake=float(row["brake"]),
                steering=float(row["steering"]),
                gear=int(row["gear"]),
                drs=int(row["drs"]),
                prompt=str(row["prompt"]),
            )
            for _, row in df.iterrows()
        ]
        return loader


# ---------------------------------------------------------------------------
# 3. F1VideoDataset
# ---------------------------------------------------------------------------

class F1VideoDataset(Dataset):
    """
    Main dataset that pairs 8-frame video clips with text prompts.

    Expected on-disk layout
    -----------------------
    data_root/
      sessions/
        {session_id}/           e.g. "2023_Monaco_VER"
          frames/
            frame_000000.png
            frame_000001.png
            ...
          telemetry.csv         (pre-exported via FastF1TelemetryLoader.save_csv)

    Each sample returned by __getitem__ is a dict:
    {
        "frames"   : FloatTensor [num_frames, 3, H, W]  normalised to [-1, 1],
        "prompt"   : str,
        "session"  : str,
        "start_idx": int,
    }

    Parameters
    ----------
    data_root   : root directory (contains sessions/ subdirectory)
    sessions    : list of session_id strings to include; None = all
    num_frames  : clip length in frames (default 8)
    image_size  : spatial resolution after resize (default 128)
    stride      : sliding-window stride (default 4)
    fps         : assumed FPS for timestamp→frame_idx conversion
    augment     : whether to apply random horizontal flipping
    """

    def __init__(
        self,
        data_root: str,
        sessions: Optional[List[str]] = None,
        num_frames: int = 8,
        image_size: int = 128,
        stride: int = 4,
        fps: float = 10.0,
        augment: bool = False,
    ) -> None:
        self.data_root  = Path(data_root)
        self.num_frames = num_frames
        self.image_size = image_size
        self.stride     = stride
        self.fps        = fps
        self.augment    = augment

        self._frame_transform = self._build_transform()
        self._index: List[Dict] = []  # list of clip metadata dicts

        self._build_index(sessions)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_transform(self) -> transforms.Compose:
        ops = [
            transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
        return transforms.Compose(ops)

    def _build_index(self, sessions: Optional[List[str]]) -> None:
        sessions_dir = self.data_root / "sessions"
        if not sessions_dir.exists():
            raise FileNotFoundError(
                f"Expected sessions directory at {sessions_dir}"
            )

        available = sorted(d.name for d in sessions_dir.iterdir() if d.is_dir())
        if sessions is not None:
            available = [s for s in available if s in sessions]

        if not available:
            raise ValueError("No matching sessions found in the dataset.")

        for session_id in available:
            session_dir  = sessions_dir / session_id
            frames_dir   = session_dir / "frames"
            telemetry_csv = session_dir / "telemetry.csv"

            if not frames_dir.exists():
                logger.warning("No frames/ dir in session %s, skipping.", session_id)
                continue

            frame_paths = sorted(frames_dir.glob("frame_*.png"))
            if len(frame_paths) < self.num_frames:
                logger.warning(
                    "Session %s has only %d frames (need %d), skipping.",
                    session_id, len(frame_paths), self.num_frames,
                )
                continue

            # Load telemetry if available
            prompts_by_frame: Optional[List[str]] = None
            if telemetry_csv.exists():
                try:
                    loader = FastF1TelemetryLoader.from_csv(str(telemetry_csv))
                    samples = loader.load()
                    prompts_by_frame = self._align_prompts_to_frames(
                        samples, len(frame_paths)
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to load telemetry for %s: %s", session_id, exc
                    )

            # Sliding-window over frames
            for start in range(
                0, len(frame_paths) - self.num_frames + 1, self.stride
            ):
                clip_paths = frame_paths[start : start + self.num_frames]

                if prompts_by_frame is not None:
                    # Use the prompt at the midpoint of the clip
                    mid = start + self.num_frames // 2
                    prompt = prompts_by_frame[min(mid, len(prompts_by_frame) - 1)]
                else:
                    prompt = "An F1 car drives on a racing circuit."

                self._index.append(
                    {
                        "session": session_id,
                        "start_idx": start,
                        "paths": [str(p) for p in clip_paths],
                        "prompt": prompt,
                    }
                )

        logger.info(
            "Dataset index built: %d clips from %d sessions.",
            len(self._index), len(available),
        )

    def _align_prompts_to_frames(
        self, samples: List[TelemetrySample], num_frames: int
    ) -> List[str]:
        """
        Map telemetry samples onto frame indices assuming uniform fps.
        Returns a list of length num_frames.
        """
        if not samples:
            return ["An F1 car drives on a racing circuit."] * num_frames

        prompts: List[str] = []
        total_duration_ms = samples[-1].timestamp_ms - samples[0].timestamp_ms
        ms_per_frame = (total_duration_ms / max(num_frames - 1, 1))

        for f_idx in range(num_frames):
            target_ms = samples[0].timestamp_ms + f_idx * ms_per_frame
            nearest = min(samples, key=lambda s: abs(s.timestamp_ms - target_ms))
            prompts.append(nearest.prompt)

        return prompts

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict:
        entry = self._index[idx]
        frames: List[torch.Tensor] = []

        for path in entry["paths"]:
            img = Image.open(path).convert("RGB")

            # Optional random horizontal flip (augmentation)
            if self.augment and random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            frames.append(self._frame_transform(img))

        frames_tensor = torch.stack(frames, dim=0)  # [F, 3, H, W]

        return {
            "frames": frames_tensor,
            "prompt": entry["prompt"],
            "session": entry["session"],
            "start_idx": entry["start_idx"],
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def extract_frames_from_video(
        video_path: str,
        output_dir: str,
        fps: float = 10.0,
        image_size: Optional[int] = None,
    ) -> int:
        """
        Helper: extract frames from an MP4/MKV into a directory of PNGs.
        Returns the number of frames written.
        """
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        sample_interval = max(1, round(src_fps / fps))
        frame_count = 0
        written = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_interval == 0:
                if image_size is not None:
                    frame = cv2.resize(
                        frame, (image_size, image_size),
                        interpolation=cv2.INTER_AREA,
                    )
                out_path = os.path.join(
                    output_dir, f"frame_{written:06d}.png"
                )
                cv2.imwrite(out_path, frame)
                written += 1

            frame_count += 1

        cap.release()
        logger.info("Extracted %d frames → %s", written, output_dir)
        return written


# ---------------------------------------------------------------------------
# 4. SyntheticF1Dataset
# ---------------------------------------------------------------------------

class SyntheticF1Dataset(Dataset):
    """
    Generates random video clips + synthetic prompts.
    Useful for unit-testing the model and training loop
    without any real footage.

    Each sample has the same structure as F1VideoDataset.__getitem__.
    """

    _SYNTHETIC_PROMPTS = [
        "The driver brakes aggressively and turns sharply right at high speed.",
        "The driver is flat out on the throttle at very high speed, DRS is open.",
        "The driver brakes moderately and turns left at medium speed.",
        "The driver partially applies throttle and steers slightly right at low speed.",
        "The driver coasts at medium speed, maintaining a straight line.",
        "The driver brakes lightly and turns right at high speed, in gear 5.",
        "The driver applies heavy throttle and steers slightly left at very high speed.",
        "The driver brakes aggressively and turns sharply left at medium speed.",
    ]

    def __init__(
        self,
        num_samples: int = 512,
        num_frames: int = 8,
        image_size: int = 128,
        seed: int = 42,
    ) -> None:
        self.num_samples = num_samples
        self.num_frames  = num_frames
        self.image_size  = image_size
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict:
        # Seed per-sample RNG for reproducibility
        torch.manual_seed(idx)

        # Random pixel video clip in [-1, 1]
        frames = torch.rand(self.num_frames, 3, self.image_size, self.image_size) * 2 - 1

        # Introduce a simple synthetic motion: global brightness drift
        for t in range(1, self.num_frames):
            drift = (t / self.num_frames) * 0.1
            frames[t] = (frames[t - 1] + drift).clamp(-1, 1)

        prompt = self._SYNTHETIC_PROMPTS[idx % len(self._SYNTHETIC_PROMPTS)]

        return {
            "frames": frames,
            "prompt": prompt,
            "session": "synthetic",
            "start_idx": idx,
        }


# ---------------------------------------------------------------------------
# 5. create_dataloader
# ---------------------------------------------------------------------------

def create_dataloader(
    dataset: Dataset,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    Factory that returns a DataLoader with sensible defaults
    for video diffusion training.

    The collate_fn stacks all tensor fields and preserves string fields
    as lists (required because DataLoader can't stack variable-length strings).
    """

    def collate_fn(batch: List[Dict]) -> Dict:
        out: Dict = {}
        for key in batch[0]:
            if isinstance(batch[0][key], torch.Tensor):
                out[key] = torch.stack([b[key] for b in batch], dim=0)
            else:
                out[key] = [b[key] for b in batch]
        return out

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )