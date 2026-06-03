"""
python preprocess.py --data_dir /workspace --output_dir /workspace/preprocessed_300
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from collections import defaultdict
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import entropy as scipy_entropy
from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import train_test_split
from tqdm import tqdm


VALID_STATIONS   = ("station1", "station2", "station3")
VALID_CONDITIONS = {1, 2, 3, 4, 5}

STATION_CROP_PARAMS = {
    "station1": {"sat_high": 25, "sat_low": 12,
                 "margins": (0.12, 0.08, 0.05, 0.02), "pad": 30, "close": 40, "open": 20},
    "station2": {"sat_high": 25, "sat_low": 12,
                 "margins": (0.10, 0.06, 0.04, 0.02), "pad": 25, "close": 35, "open": 18},
    "station3": {"sat_high": 20, "sat_low": 10,
                 "margins": (0.10, 0.06, 0.04, 0.02), "pad": 30, "close": 40, "open": 20},
}
DEFAULT_CROP_PARAMS = STATION_CROP_PARAMS["station1"]

QUALITY_THRESHOLDS = {
    "laplacian_min": 95.0,
    "tenengrad_min": 2300.0,
}

HAMMING_THRESHOLD = 10
SSIM_THRESHOLD    = 0.90
SEVERE_BLUR_PCT   = 0.10

MANIFEST_COLS = [
    "station", "month", "item_id", "condition", "split",
    "front_path", "back_path", "front_src", "back_src", "json_path",
]


def setup_logging(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "preprocess.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("preprocess")


def read_condition(json_path):
    try:
        val = int(json.loads(json_path.read_text(encoding="utf-8")).get("condition"))
        return val if val in VALID_CONDITIONS else None
    except Exception:
        return None


def collect_items(data_dir):
    rows = []
    for station in VALID_STATIONS:
        station_dir = data_dir / station
        if not station_dir.exists():
            continue
        for month_dir in sorted(p for p in station_dir.iterdir() if p.is_dir()):
            for json_path in sorted(month_dir.glob("labels_*.json")):
                item_id   = json_path.stem.replace("labels_", "")
                front_src = month_dir / f"front_{item_id}.jpg"
                back_src  = month_dir / f"back_{item_id}.jpg"
                rows.append({
                    "station":   station,
                    "month":     month_dir.name,
                    "item_id":   item_id,
                    "condition": read_condition(json_path),
                    "front_src": str(front_src),
                    "back_src":  str(back_src),
                    "json_path": str(json_path),
                    "has_front": front_src.exists(),
                    "has_back":  back_src.exists(),
                })
    return pd.DataFrame(rows)


def _parse_meta(json_path):
    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except Exception:
        return {k: "unknown" for k in ("brand", "type_", "category", "colors", "size")}
    colors = data.get("colors", [])
    colors_str = "|".join(sorted(str(c).lower().strip() for c in colors)) if isinstance(colors, list) else str(colors).lower().strip()
    return {
        "brand":    str(data.get("brand",    "")).lower().strip(),
        "type_":    str(data.get("type",     "")).lower().strip(),
        "category": str(data.get("category", "")).lower().strip(),
        "colors":   colors_str,
        "size":     str(data.get("size",     "")).lower().strip(),
    }


def _group_key(row):
    return f"{row['brand']}|{row['type_']}|{row['category']}|{row['colors']}|{row['size']}"


def _phash(path):
    try:
        return imagehash.phash(Image.open(path).convert("RGB"))
    except Exception:
        return None


def _ssim(p1, p2, size=(256, 256)):
    try:
        g1 = np.array(Image.open(p1).resize(size).convert("L"))
        g2 = np.array(Image.open(p2).resize(size).convert("L"))
        score, _ = ssim(g1, g2, full=True)
        return float(score)
    except Exception:
        return 0.0


def detect_duplicates(df, logger):
    logger.info("중복 탐지 시작")

    metas = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="메타 파싱", leave=False):
        m = _parse_meta(row["json_path"])
        m.update({"item_id": row["item_id"], "station": row["station"],
                  "month": row["month"], "front_src": row["front_src"]})
        metas.append(m)
    meta_df = pd.DataFrame(metas)

    groups = defaultdict(list)
    for _, row in meta_df.iterrows():
        groups[_group_key(row)].append(row.to_dict())
    candidates = {k: v for k, v in groups.items() if len(v) > 1}
    candidate_items = [it for g in candidates.values() for it in g]
    logger.info(f"  후보 그룹 {len(candidates):,}개, 아이템 {len(candidate_items):,}개")

    for it in tqdm(candidate_items, desc="pHash 계산", leave=False):
        it["phash"] = _phash(it["front_src"])

    phash_pairs = []
    for group_items in candidates.values():
        valid = [it for it in group_items if it.get("phash") is not None]
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                dist = valid[i]["phash"] - valid[j]["phash"]
                if dist <= HAMMING_THRESHOLD:
                    phash_pairs.append((valid[i], valid[j]))
    logger.info(f"  pHash 유사 쌍: {len(phash_pairs):,}")

    dup_ids = set()
    for it1, it2 in tqdm(phash_pairs, desc="SSIM 검증", leave=False):
        if _ssim(it1["front_src"], it2["front_src"]) >= SSIM_THRESHOLD:
            remove = it1["item_id"] if it1["item_id"] > it2["item_id"] else it2["item_id"]
            dup_ids.add(remove)

    logger.info(f"  중복 제거: {len(dup_ids):,}개")
    return dup_ids


def _read_img(path):
    # cv2.imread는 한글 경로를 읽지 못해 fromfile로 우회
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def _laplacian(img):
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

def _tenengrad(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.Sobel(g, cv2.CV_64F, 1, 0)**2 + cv2.Sobel(g, cv2.CV_64F, 0, 1)**2))


def detect_blur(df, logger):
    logger.info("품질 필터링 시작")

    records = []
    t = QUALITY_THRESHOLDS
    for _, row in tqdm(df.iterrows(), total=len(df), desc="품질 스캔", leave=False):
        for col in ("front_src", "back_src"):
            img = _read_img(row[col])
            if img is None:
                continue
            lap, ten = _laplacian(img), _tenengrad(img)
            records.append({
                "item_id":   row["item_id"],
                "laplacian": lap,
                "tenengrad": ten,
                "flag_blur": lap < t["laplacian_min"] or ten < t["tenengrad_min"],
            })

    qdf = pd.DataFrame(records)
    blur_df = qdf[qdf["flag_blur"]].copy()

    if len(blur_df) == 0:
        logger.info("  흐림 이미지 없음")
        return set()

    lap_cut = blur_df["laplacian"].quantile(SEVERE_BLUR_PCT)
    ten_cut = blur_df["tenengrad"].quantile(SEVERE_BLUR_PCT)
    severe_ids = set(
        blur_df[(blur_df["laplacian"] <= lap_cut) & (blur_df["tenengrad"] <= ten_cut)]["item_id"].unique()
    )

    logger.info(f"  전체 흐림 {blur_df['item_id'].nunique():,}개, 심한 흐림 제거 {len(severe_ids):,}개")
    return severe_ids


def _try_crop(img, sat_thresh, params):
    h, w = img.shape[:2]
    _, s, _ = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    _, mask = cv2.threshold(cv2.GaussianBlur(s, (31, 31), 0), sat_thresh, 255, cv2.THRESH_BINARY)

    lm, bm, rm, tm = params["margins"]
    edge = np.ones_like(mask, dtype=np.uint8) * 255
    edge[:, :int(w*lm)] = 0
    edge[int(h*(1-bm)):, :] = 0
    edge[:, int(w*(1-rm)):] = 0
    edge[:int(h*tm), :] = 0
    mask = cv2.bitwise_and(mask, edge)

    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params["close"],)*2)
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params["open"],)*2)
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc), cv2.MORPH_OPEN, ko)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < h * w * 0.03:
        return None

    x, y, cw, ch = cv2.boundingRect(largest)
    p = params["pad"]
    return img[max(0,y-p):min(h,y+ch+p), max(0,x-p):min(w,x+cw+p)]


def crop_garment(img, station="station1"):
    params = STATION_CROP_PARAMS.get(station, DEFAULT_CROP_PARAMS)
    for sat in (params["sat_high"], params["sat_low"]):
        result = _try_crop(img, sat, params)
        if result is not None:
            return result
    h, w = img.shape[:2]
    mx, my = int(w*0.15), int(h*0.15)
    return img[my:h-my, mx:w-mx]


def resize_pad(img, target=300):
    h, w = img.shape[:2]
    scale = target / max(h, w)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((target, target, 3), 255, dtype=np.uint8)
    canvas[(target-nh)//2:(target-nh)//2+nh, (target-nw)//2:(target-nw)//2+nw] = resized
    return canvas


def preprocess_images(df, output_dir, target_size, resume, logger):
    logger.info(f"이미지 전처리 시작 (target={target_size}px)")

    front_paths, back_paths = [], []
    done = skipped = errors = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Crop+Resize", leave=False):
        dst_dir = output_dir / row["station"] / row["month"]
        dst_dir.mkdir(parents=True, exist_ok=True)

        front_dst = dst_dir / f"front_{row['item_id']}.jpg"
        back_dst  = dst_dir / f"back_{row['item_id']}.jpg"
        front_paths.append(str(front_dst))
        back_paths.append(str(back_dst))

        if resume and front_dst.exists() and back_dst.exists():
            skipped += 1
            continue

        ok = True
        for src, dst in [(row["front_src"], front_dst), (row["back_src"], back_dst)]:
            img = _read_img(src)
            if img is None:
                errors += 1
                ok = False
                continue
            buf_ok, buf = cv2.imencode(".jpg", resize_pad(crop_garment(img, row["station"]), target_size),
                                       [cv2.IMWRITE_JPEG_QUALITY, 95])
            if buf_ok:
                buf.tofile(str(dst))

        json_src = Path(row["json_path"])
        json_dst = dst_dir / json_src.name
        if not json_dst.exists():
            shutil.copy2(str(json_src), str(json_dst))

        done += ok

    logger.info(f"  완료 {done:,}  스킵 {skipped:,}  에러 {errors:,}")

    df = df.copy()
    df["front_path"] = front_paths
    df["back_path"]  = back_paths
    return df


def make_splits(df, seed, logger):
    logger.info("Train/Val/Test 분할")
    train_df, tmp_df = train_test_split(df, test_size=0.30, random_state=seed,
                                        shuffle=True, stratify=df["condition"])
    val_df, test_df  = train_test_split(tmp_df, test_size=1/3, random_state=seed,
                                        shuffle=True, stratify=tmp_df["condition"])
    train_df = train_df.copy(); train_df["split"] = "train"
    val_df   = val_df.copy();   val_df["split"]   = "val"
    test_df  = test_df.copy();  test_df["split"]  = "test"
    result = pd.concat([train_df, val_df, test_df], ignore_index=True)
    for split, cnt in result["split"].value_counts().sort_index().items():
        logger.info(f"  {split}: {cnt:,}")
    return result


def save_manifests(df, manifest_dir, policy, logger):
    manifest_dir.mkdir(parents=True, exist_ok=True)
    df[MANIFEST_COLS].to_csv(manifest_dir / "clean_items_with_splits.csv",
                              index=False, encoding="utf-8-sig")
    for split in ("train", "val", "test"):
        df.loc[df["split"] == split, MANIFEST_COLS].to_csv(
            manifest_dir / f"{split}.csv", index=False, encoding="utf-8-sig")
    (manifest_dir / "cleaning_policy.json").write_text(
        json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"manifest 저장 완료 → {manifest_dir}")


def run(args):
    data_dir     = Path(args.data_dir).resolve()
    output_dir   = Path(args.output_dir).resolve()
    manifest_dir = output_dir / "manifests"
    logger = setup_logging(output_dir)
    t0 = time.time()

    logger.info(f"data_dir={data_dir}  output_dir={output_dir}  target={args.target_size}px")

    all_df = collect_items(data_dir)
    logger.info(f"전체 아이템 {len(all_df):,}개")

    valid_df = all_df[all_df["has_front"] & all_df["has_back"] & all_df["condition"].notna()].copy()
    valid_df["condition"] = valid_df["condition"].astype(int)
    logger.info(f"유효 아이템 {len(valid_df):,}개")

    dup_ids  = detect_duplicates(valid_df, logger) if not args.skip_dedup  else set()
    blur_ids = detect_blur(valid_df, logger)       if not args.skip_quality else set()

    exclude = dup_ids | blur_ids
    clean_df = valid_df[~valid_df["item_id"].isin(exclude)].copy().reset_index(drop=True)
    logger.info(f"제외 {len(exclude):,}개 → 사용 {len(clean_df):,}개")

    processed_df = preprocess_images(clean_df, output_dir, args.target_size, args.resume, logger)
    split_df = make_splits(processed_df, args.seed, logger)

    policy = {
        "raw_items":        int(len(all_df)),
        "valid_items":      int(len(valid_df)),
        "dup_excluded":     int(len(dup_ids)),
        "blur_excluded":    int(len(blur_ids)),
        "clean_items":      int(len(clean_df)),
        "target_size":      args.target_size,
        "seed":             args.seed,
        "condition_counts": {str(k): int(v) for k, v in
                             clean_df["condition"].value_counts().sort_index().items()},
        "split_counts":     {s: int((split_df["split"] == s).sum()) for s in ("train","val","test")},
    }
    save_manifests(split_df, manifest_dir, policy, logger)

    logger.info(f"완료 ({time.time()-t0:.1f}초)  결과: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="원본 데이터 → 전처리 완료 데이터셋 생성")
    parser.add_argument("--data_dir",     required=True)
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--target_size",  type=int, default=300)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--skip_dedup",   action="store_true")
    parser.add_argument("--skip_quality", action="store_true")
    parser.add_argument("--resume",       action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
