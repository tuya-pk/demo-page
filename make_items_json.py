import json
from pathlib import Path
import pandas as pd   # ← 新增

DEMO_ROOT = Path(".")
WAV_DIR = DEMO_ROOT / "wav"
ORIG_DIR = WAV_DIR / "orig"
LM_DIR = WAV_DIR / "lm"
WOLM_DIR = WAV_DIR / "without_lm"
SOUNDSTREAM_DIR =  WAV_DIR / "soundstream"
LOSS_RATES = ["0.1", "0.2", "0.3"]  # 你要展示的三个loss rate
AUDIO_EXT = ".wav"

CSV_DIR = Path("../res2_save_dir")
REF_CSV = "0.3_per_utt.csv"   # 用哪个 csv 提供 ref
REF_CANDIDATES = ["ref_norm_ours", "ref_norm", "ref_text"]

def main():
    if not ORIG_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {ORIG_DIR}")

    # 以 wav/orig 里的文件名为准：<utt_id>.wav
    utt_ids = sorted([p.stem for p in ORIG_DIR.glob(f"*{AUDIO_EXT}")])

    if not utt_ids:
        raise RuntimeError(f"No wav files found in {ORIG_DIR}")

    items = []
    missing = []
    ref_map = {}
    df = pd.read_csv(CSV_DIR / REF_CSV)
    df["utt_id"] = df["utt_id"].astype(str)
    ref_col = None
    for c in REF_CANDIDATES:
        if c in df.columns:
            ref_col = c
            break

    if ref_col is None:
        raise RuntimeError(f"No ref column found in {REF_CSV}")
    for _, row in df.iterrows():
        uid = row["utt_id"]
        ref_map[uid] = str(row[ref_col])

    for uid in utt_ids:
        entry = {
            "utt_id": uid,
            "orig_audio": f"./wav/orig/{uid}{AUDIO_EXT}",
            "losses": {}
        }

        for lr in LOSS_RATES:
            lm_name = f"{lr}_{uid}{AUDIO_EXT}"
            wolm_name = f"{lr}_{uid}{AUDIO_EXT}"
            soundstream_name = f"{lr}_{uid}{AUDIO_EXT}"

            lm_path = LM_DIR / lm_name
            wolm_path = WOLM_DIR / wolm_name

            if not lm_path.exists():
                missing.append(str(lm_path))
            if not wolm_path.exists():
                missing.append(str(wolm_path))

            entry["losses"][lr] = {
                "ref": ref_map.get(uid, ""),  # 你页面要显示ref的话，下一步我们再从csv填充
                "audio_lm": f"./wav/lm/{lm_name}",
                "audio_wolm": f"./wav/without_lm/{wolm_name}",
                "audio_ss": f"./wav/soundstream/{soundstream_name}"
            }

        items.append(entry)

    out = {"loss_rates": LOSS_RATES, "items": items}

    Path("items.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Wrote items.json with {len(items)} items.")
    if missing:
        print(f"[WARN] Missing {len(missing)} files:")
        for x in missing[:30]:
            print("  ", x)
        if len(missing) > 30:
            print("  ...")

if __name__ == "__main__":
    main()
