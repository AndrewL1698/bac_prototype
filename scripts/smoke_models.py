"""Model smoke tests. Each check prints PASS/FAIL with timing; failures never abort the script.

    python scripts/run_job.py --name smoke -- python scripts/smoke_models.py [--only clip,aesthetic,...]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from montage import paths  # noqa: E402
from montage.config import get_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS: dict[str, str] = {}


def check(name: str):
    def deco(fn):
        def run():
            t0 = time.time()
            try:
                detail = fn() or ""
                RESULTS[name] = "PASS"
                print(f"PASS {name} ({time.time() - t0:.1f}s) {detail}", flush=True)
            except Exception as e:  # noqa: BLE001
                RESULTS[name] = f"FAIL: {e}"
                print(f"FAIL {name} ({time.time() - t0:.1f}s): {e}", flush=True)
                traceback.print_exc(limit=3)
        run.__name__ = name
        return run
    return deco


def _two_images():
    from PIL import Image, ImageDraw
    red = Image.new("RGB", (224, 224), "white")
    ImageDraw.Draw(red).rectangle([50, 50, 174, 174], fill="red")
    blue = Image.new("RGB", (224, 224), "white")
    ImageDraw.Draw(blue).ellipse([50, 50, 174, 174], fill="blue")
    return red, blue


@check("clip")
def clip():
    from montage.features import clip_encoder
    ims = clip_encoder.encode_images(_two_images())
    txt = clip_encoder.encode_text(["a red square", "a blue circle"])
    sim = ims @ txt.T
    assert sim[0, 0] > sim[0, 1] and sim[1, 1] > sim[1, 0], f"sim matrix {sim.round(3).tolist()}"
    return f"sim diag {sim[0, 0]:.3f}/{sim[1, 1]:.3f} device {clip_encoder.device()}"


@check("aesthetic")
def aesthetic():
    from montage.features import aesthetic as aes
    from montage.features import clip_encoder
    assert aes.get_head() is not None, "weights not loaded"
    s = aes.score(clip_encoder.encode_images(_two_images()))
    assert all(1.0 < v < 9.5 for v in s), f"scores {s}"
    return f"scores {[round(float(v), 2) for v in s]}"


@check("mert")
def mert():
    import numpy as np
    import torch
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    cfg = get_config().music.emotion
    model = AutoModel.from_pretrained(cfg.model, trust_remote_code=True, revision=cfg.revision).eval()
    fe = Wav2Vec2FeatureExtractor.from_pretrained(cfg.model, trust_remote_code=True, revision=cfg.revision)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    sr = fe.sampling_rate
    x = fe(np.random.randn(5 * sr).astype(np.float32) * 0.1, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        out = model(**{k: v.to(dev) for k, v in x.items()}, output_hidden_states=True)
    hs = out.hidden_states
    assert len(hs) == 13 and hs[-1].shape[-1] == 768, f"{len(hs)} states, shape {tuple(hs[-1].shape)}"
    import transformers
    return f"13 x {tuple(hs[-1].shape)} transformers {transformers.__version__}"


@check("beat_this")
def beat_this():
    import numpy as np
    from beat_this.inference import File2Beats
    fx = REPO / "tests" / "fixtures"
    truth = json.loads((fx / "truth.json").read_text())["click_120.wav"]
    import torch
    f2b = File2Beats(checkpoint_path="final0", device="cuda" if torch.cuda.is_available() else "cpu", dbn=False)
    beats, downbeats = f2b(str(fx / "click_120.wav"))
    tb = np.array(truth["beats"])
    hit = np.mean([np.min(np.abs(tb - b)) < 0.03 for b in beats]) if len(beats) else 0
    assert hit >= 0.95 and len(downbeats) > 0, f"hit {hit:.2f}, {len(downbeats)} downbeats"
    return f"{len(beats)} beats, {hit:.0%} within 30 ms, {len(downbeats)} downbeats"


@check("yunet")
def yunet():
    import cv2
    import numpy as np
    from montage import weights
    det = cv2.FaceDetectorYN.create(str(weights.yunet_path()), "", (320, 320), get_config().features.face_score)
    img = np.full((320, 320, 3), 128, np.uint8)
    det.setInputSize((320, 320))
    _, faces = det.detect(img)
    return f"ran ok, {0 if faces is None else len(faces)} faces on blank image, cv2 {cv2.__version__}"


@check("claude")
def claude():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return "skipped (no ANTHROPIC_API_KEY)"
    import anthropic
    from pydantic import BaseModel

    class Mini(BaseModel):
        vibe: str
        pacing: float

    client = anthropic.Anthropic()
    r = client.messages.parse(
        model=get_config().director.model, max_tokens=500,
        messages=[{"role": "user", "content": "Brief: 'chill beach recap, slow'. Return vibe and pacing in [-1,1]."}],
        output_format=Mini,
    )
    return f"parsed {r.parsed_output}"


@check("ffmpeg")
def ffmpeg():
    cfg = get_config().render
    ff = cfg.ffmpeg_bin
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "t.mp4"
        subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30:duration=1",
                        "-c:v", "libx264", "-preset", cfg.preset, "-crf", str(cfg.preview.crf), "-pix_fmt", "yuv420p",
                        "-r", str(cfg.fps), str(out)], check=True)
        n = subprocess.run([cfg.ffprobe_bin, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                            "stream=nb_read_frames", "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout
        assert int(n.strip()) == 30, f"frames {n!r}"
    filters = subprocess.run([ff, "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    zs = any(" zscale " in ln for ln in filters.splitlines())
    tm = any(" tonemap " in ln for ln in filters.splitlines())
    return f"30 frames ok; zscale {'yes' if zs else 'no'}, tonemap {'yes' if tm else 'no'}"


CHECKS = [clip, aesthetic, mert, beat_this, yunet, claude, ffmpeg]


def main(argv: list[str] | None = None) -> int:
    paths.load_colab_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args(argv)
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    for i, c in enumerate(CHECKS):
        if only and c.__name__ not in only:
            continue
        try:
            from montage import jobs
            jobs.report(i / len(CHECKS), f"checking {c.__name__}", force=True)
        except Exception:
            pass
        c()
    print("SUMMARY " + json.dumps(RESULTS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
