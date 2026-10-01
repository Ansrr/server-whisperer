"""Turn server health into sound for The Server Whisperer."""
import io
import wave

import numpy as np


def sonify(df, if_threshold, view_minutes=900, seconds=20, sr=22050):
    """Calm hum when healthy, clashing notes as it drifts, beeping when an alert fires."""
    start = len(df) - view_minutes
    lo = float(np.median(df["iso"].iloc[:1000]))
    level = np.clip((df["iso"].values[start:] - lo) / max(if_threshold - lo, 1e-6), 0, 1)
    alert = df["model_alert"].values[start:].astype(float)

    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    x = np.linspace(0, seconds, len(level))
    lvl = np.interp(t, x, level)
    alr = np.interp(t, x, alert)

    base = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    clash = lvl * (0.35 * np.sin(2 * np.pi * 311.1 * t) + 0.3 * np.sin(2 * np.pi * 232.0 * t))
    trem_phase = 2 * np.pi * np.cumsum(2.0 + 8.0 * lvl) / sr
    tremolo = 1.0 + 0.4 * lvl * np.sin(trem_phase)
    beep = alr * 0.3 * np.sin(2 * np.pi * 880.0 * t) * (np.sin(2 * np.pi * 4.0 * t) > 0)

    sig = (base + clash) * tremolo + beep
    fade = np.minimum(1.0, np.minimum(t, seconds - t) / 0.5)
    sig = sig * fade
    sig = 0.8 * sig / max(np.max(np.abs(sig)), 1e-6)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((sig * 32767).astype(np.int16).tobytes())
    return buf.getvalue()
