"""In-distribution-style retain pool: text-to-image with SD-Turbo in the dataset's 'cinematic' look (warm light,
shallow depth of field), for ~1250 object/scene names; then the ORIGINAL model sorts the images into its 100 classes
(only confidently classified ones are used, see common.load_genpool).

Why: the training data are AI-generated cinematic images (inspect_montage.jpg), while mini-ImageNet photos only reach
61/100 classes and are often unrelated to the class they land in -> anchors / SSD F_r computed on them leave ~40
classes unprotected (SSD class collapse: 0.048 on the server).

Run:
  1) login node:  HF_HOME=<hf cache dir> python data_prep/gen_pool.py --check
     (add --download if SD-Turbo is not cached yet)
  2) sbatch genpool.sh        (4 GPUs ~10 min: shards, then --merge = classify + coverage stats + preview)
Output: ../output/genpool.npz (images uint8 [M,224,224,3], label = original model's class, prob = its confidence,
        prompt = prompt index), ../output/genpool_prompts.json, ../output/genpool_preview.jpg
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules (common, diagnostics, ...)

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import OUTPUT_DIR, load_forget_images, load_model, predict

GEN_DIR = Path(os.environ.get("GEN_TMP", OUTPUT_DIR / "gen"))  # shards (temporary): $HOME when scratch inodes are full
POOL_FILE = OUTPUT_DIR / "genpool.npz"
TPOOL_FILE = OUTPUT_DIR / "genpool_t.npz"  # --targeted round: more images of the names each class already attracts
PPOOL_FILE = OUTPUT_DIR / "genpool_p.npz"  # --targeted --photo: same names, plain photographic look (lower ext drift)
DEFAULT_MODEL = "stabilityai/sd-turbo"
STYLES = [
    "cinematic photo of {}, golden hour, warm soft light, shallow depth of field, bokeh, film grain",
    "a moody cinematic still of {}, dramatic lighting, 35mm photograph, highly detailed",
    "{}, atmospheric cinematic photograph, natural window light, depth of field, high quality",
]
PHOTO_STYLES = [
    "a photo of {}, daylight, natural colors, sharp focus",
    "a candid snapshot of {}, everyday scene, smartphone photo",
    "{}, raw photo, neutral lighting, realistic, high detail",
]
EXTRA = """lantern, hanging lantern, stone garden lantern, oil lamp, candle, candle holder, porcelain figurine,
ceramic fox figurine, ceramic cat figurine, small statue, bronze sculpture, vase, teapot, teacup, coffee mug,
wine glass, glass bottle, mason jar, ceramic bowl, dinner plate, wicker basket, knitted blanket, pillow, armchair,
wooden chair, rocking chair, office chair, garden chair, sofa, bed, wooden table, desk, bookshelf, old book,
open book, notebook and pen, typewriter, wall clock, alarm clock, pocket watch, mirror, window with curtains,
wooden door, staircase, fireplace, rug, desk lamp, chandelier, potted plant, flowers in a vase, bouquet of roses,
rowboat, canoe, kayak, sailboat, yacht, fishing boat, cargo ship, ferry, rowing on a lake, small boat on a calm lake,
wooden pier, dock, lighthouse, harbor, sandy beach, ocean waves, misty lake, river, waterfall, mountain peak,
pine forest, meadow, desert dunes, snowy landscape, autumn trees, cherry blossom tree, sunflower field, wheat field,
vineyard, garden, park bench, street lamp, city street at night, alley, bridge, train station, railway tracks,
old train, tram, bus, taxi, vintage car, motorcycle, bicycle, scooter, airplane, hot air balloon, helicopter,
tractor, truck, camper van, tent, campfire, cabin in the woods, farmhouse, barn, windmill, castle, church,
temple, pagoda, skyscraper, apartment building, cafe, restaurant table, bakery, market stall, bookstore, library,
kitchen, bathroom, bedroom, living room, attic, greenhouse, workshop tools, hammer, guitar, violin, piano,
drum, record player, vintage radio, camera, telephone, laptop, smartphone, headphones, television, umbrella,
hat, backpack, suitcase, shoes, boots, dress, coat, scarf, gloves, necklace, ring, wristwatch, sunglasses,
toy car, teddy bear, doll, puzzle, chess board, playing cards, dice, kite, balloon, globe, map, compass,
telescope, microscope, hourglass, key, padlock, birdcage, bird house, mailbox, fountain, swing, ladder,
wheelbarrow, watering can, bucket, broom, fruit bowl, apples, oranges, lemons, strawberries, bread loaf,
croissant, cake, cupcake, pizza, pasta, soup, sushi, burger, ice cream, chocolate, honey jar, cheese, coffee beans,
tea leaves, wine bottle, cocktail, dog, puppy, cat, kitten, horse, cow, sheep, goat, pig, chicken, duck, swan,
goose, owl, eagle, parrot, sparrow, hummingbird, butterfly, bee, ladybug, dragonfly, spider web, snail, frog,
turtle, lizard, snake, fish in an aquarium, goldfish, jellyfish, dolphin, whale, seal, penguin, polar bear,
brown bear, deer, fox, wolf, rabbit, squirrel, hedgehog, raccoon, mouse, lion, tiger, elephant, giraffe, zebra,
monkey, panda, koala, kangaroo, camel, an old man reading, a woman drinking coffee, a child playing, hands holding
a cup, a person walking in the rain, a fisherman, a painter at an easel, a chef cooking, a musician playing,
mushrooms in the forest, autumn leaves, rain on a window, foggy road, sunset over the sea, starry night sky,
full moon, thunderstorm, rainbow, snowflakes, ice, rocks on a beach, seashells, coral reef, cave, volcano""".replace(
    "\n", " ")


def nouns():
    from timm.data import ImageNetInfo
    info = ImageNetInfo()
    names = [info.index_to_description(i).split(",")[0].strip() for i in range(info.num_classes())]
    names += [n.strip() for n in EXTRA.split(",") if n.strip()]
    seen, out = set(), []
    for n in names:  # dedupe, keep order
        if n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out


def prompts():
    return [s.format(("a " if not n.lower().startswith(("a ", "an ")) else "") + n) for n in nouns() for s in STYLES]


def targeted_prompts(names_per_class=6, target=300, styles=None):
    """Round 2: for every class, the names whose round-1 images the original model put there most confidently
    (count x mean prob, >= 2 confident images); enough seeds per (name, style) for ~`target` images per class.
    Deterministic (same list in every shard). -> list of (prompt, n_images)."""
    with np.load(POOL_FILE, allow_pickle=False) as d:
        label, prob, pidx = d["label"], d["prob"], d["prompt"]
    names = nouns()
    noun = pidx // len(STYLES)
    ok = prob >= 0.5
    out = []
    for c in range(100):
        m = ok & (label == c)
        scores = []
        for n in np.unique(noun[m]):
            k = int((m & (noun == n)).sum())
            if k >= 2:
                scores.append((k * float(prob[m & (noun == n)].mean()), int(n)))
        top = [n for _, n in sorted(scores, reverse=True)[:names_per_class]]
        if not top:
            continue
        styles = styles or STYLES
        per = int(np.ceil(target / (len(top) * len(styles))))
        for n in top:
            name = names[n]
            for st in styles:
                out.append((st.format(("a " if not name.lower().startswith(("a ", "an ")) else "") + name), per))
    return out


def make_pipe(model, device):
    from diffusers import AutoPipelineForText2Image
    kw = dict(torch_dtype=torch.float16, variant="fp16") if device == "cuda" else {}
    pipe = AutoPipelineForText2Image.from_pretrained(model, safety_checker=None, requires_safety_checker=False, **kw)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def run_shard(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = make_pipe(args.model, device)
    if args.targeted:
        T = targeted_prompts(styles=PHOTO_STYLES if args.photo else None)
        P = [t for t, _ in T]
        all_jobs = [(p, k) for p, (_, n) in enumerate(T) for k in range(n)]
    else:
        P = prompts()
        all_jobs = [(p, k) for p in range(len(P)) for k in range(args.per_prompt)]
    if args.limit:  # quick tests
        all_jobs = all_jobs[:args.limit]
    jobs = [all_jobs[i] for i in np.array_split(np.arange(len(all_jobs)), args.nshards)[args.shard]]
    t0, imgs, pidx = time.time(), [], []
    for i in range(0, len(jobs), args.batch_size):
        batch = jobs[i:i + args.batch_size]
        g = torch.Generator(device=device).manual_seed(args.seed + 1000003 * args.shard + i + 7777 * args.targeted + 31337 * args.photo)
        res = pipe(prompt=[P[p] for p, _ in batch], num_inference_steps=args.steps, guidance_scale=0.0,
                   height=args.size, width=args.size, generator=g, output_type="pt").images
        res = F.interpolate(res.float(), size=(224, 224), mode="bicubic", antialias=True, align_corners=False)
        imgs.append((res.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy())
        pidx += [p for p, _ in batch]
        if (i // args.batch_size) % 20 == 0:
            print(f"shard {args.shard}: {i + len(batch)}/{len(jobs)} images ({time.time() - t0:.0f}s)", flush=True)
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    out = GEN_DIR / f"pool{args.shard}.npz"
    tmp = GEN_DIR / f"pool{args.shard}.tmp{os.getpid()}.npz"
    np.savez(tmp, images=np.concatenate(imgs), prompt=np.array(pidx, dtype=np.int64))
    os.replace(tmp, out)
    print(f"shard {args.shard}: saved {out} ({time.time() - t0:.0f}s)")


def thumb(a, size):
    return Image.fromarray(np.asarray(a)).resize((size, size))


def merge(args):
    files = [f for f in sorted(GEN_DIR.glob("pool[0-9]*.npz")) if ".tmp" not in f.name]
    assert files, f"no shards in {GEN_DIR}"
    imgs, pidx = [], []
    for f in files:
        with np.load(f, allow_pickle=False) as d:
            imgs.append(d["images"])
            pidx.append(d["prompt"])
    imgs, pidx = np.concatenate(imgs), np.concatenate(pidx)
    teacher = load_model()
    probs = F.softmax(predict(teacher, torch.from_numpy(imgs)).float(), 1)
    prob, label = probs.max(1)
    forget = load_forget_images()
    fl = predict(teacher, forget).argmax(1)
    fcls = torch.bincount(fl, minlength=100) > 0
    out = (PPOOL_FILE if args.photo else TPOOL_FILE) if args.targeted else POOL_FILE
    tmp = out.with_name(out.stem + f".tmp{os.getpid()}.npz")
    np.savez(tmp, images=imgs, label=label.numpy(), prob=prob.numpy(), prompt=pidx)
    os.replace(tmp, out)
    P = ([t for t, _ in targeted_prompts(styles=PHOTO_STYLES if args.photo else None)] if args.targeted
         else prompts())
    (GEN_DIR / f"{out.stem}_prompts.json").write_text(json.dumps(P))
    print(f"saved {out}: {imgs.shape} ({out.stat().st_size / 1e9:.2f} GB)")
    for thr in (0.5, 0.8):
        cnt = torch.bincount(label[prob >= thr], minlength=100)
        print(f"prob >= {thr}: {int((prob >= thr).sum())} images; classes with >=10 images: {int((cnt >= 10).sum())}"
              f"/100 (forget classes {int(((cnt >= 10) & fcls).sum())}/{int(fcls.sum())}, untouched "
              f"{int(((cnt >= 10) & ~fcls).sum())}/{int((~fcls).sum())}); >=50 images: {int((cnt >= 50).sum())}/100")
    if args.targeted:  # coverage of round 1 + round 2 together
        with np.load(POOL_FILE, allow_pickle=False) as d:
            l1, p1 = torch.from_numpy(d["label"]), torch.from_numpy(d["prob"])
        cnt = torch.bincount(l1[p1 >= 0.5], minlength=100) + torch.bincount(label[prob >= 0.5], minlength=100)
        print(f"round1+2, prob >= 0.5: classes with >=50 images: {int((cnt >= 50).sum())}/100, >=150: "
              f"{int((cnt >= 150).sum())}/100; weakest forget classes: "
              + ", ".join(f"{c}:{int(cnt[c])}" for c in (cnt + 10**6 * ~fcls).argsort()[:8].tolist()))
        for f in files:
            f.unlink()
        return
    # preview: forget classes (5 forget images | 6 generated), then untouched classes (11 generated)
    size, cols = 96, 12
    cnt = torch.bincount(label[prob >= 0.5], minlength=100)
    rows = []
    for c in (cnt * fcls).argsort(descending=True)[:6].tolist():
        fi = (fl == c).nonzero().flatten()[:5].tolist()
        gi = ((label == c) & (prob >= 0.5)).nonzero().flatten()[:6].tolist()
        rows.append([forget[i].numpy() for i in fi] + [None] + [imgs[i] for i in gi])
    for c in (cnt * ~fcls).argsort(descending=True)[:6].tolist():
        gi = ((label == c) & (prob >= 0.5)).nonzero().flatten()[:11].tolist()
        rows.append([imgs[i] for i in gi])
    canvas = Image.new("RGB", (cols * size, len(rows) * size), "white")
    for r, row in enumerate(rows):
        for k, a in enumerate(row[:cols]):
            if a is not None:
                canvas.paste(thumb(a, size), (k * size, r * size))
    canvas.save(OUTPUT_DIR / "genpool_preview.jpg", quality=85)
    print(f"preview: {OUTPUT_DIR / 'genpool_preview.jpg'} (rows 1-6: forget class = 5 forget images | generated; "
          f"rows 7-12: untouched classes, generated only)")
    for f in files:
        f.unlink()


def check(args):
    P = prompts()
    print(f"{len(nouns())} names x {len(STYLES)} styles = {len(P)} prompts; x {args.per_prompt} = "
          f"{len(P) * args.per_prompt} images")
    print("example:", P[0], "|", P[-1])
    try:
        import diffusers
        print("diffusers", diffusers.__version__)
    except ImportError:
        print("diffusers NOT installed in this venv")
    from huggingface_hub import constants, snapshot_download
    print(f"HF cache dir: {constants.HF_HUB_CACHE}")
    pats = ["model_index.json", "*/*.json", "*/*.txt", "*.fp16.safetensors", "*/*.fp16.safetensors"]
    if args.download:
        snapshot_download(args.model, allow_patterns=pats)
    try:
        path = Path(snapshot_download(args.model, allow_patterns=pats, local_files_only=True))
    except Exception as exc:  # noqa: BLE001
        print(f"{args.model} NOT cached ({type(exc).__name__}); rerun with --download")
        return
    weights = {str(f.relative_to(path)): f.stat().st_size / 1e9 for f in path.rglob("*.safetensors")}
    print(f"{args.model} at {path}: weights {', '.join(f'{k} {v:.2f} GB' for k, v in weights.items()) or 'NONE'}")
    need = ("unet/diffusion_pytorch_model.fp16.safetensors", "text_encoder/model.fp16.safetensors",
            "vae/diffusion_pytorch_model.fp16.safetensors")
    missing = [n for n in need if n not in weights]
    print("READY for sbatch genpool.sh" if not missing else f"MISSING {missing}: rerun with --download")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="login node: prompts, diffusers, model cache")
    ap.add_argument("--download", action="store_true", help="with --check: download SD-Turbo")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--targeted", action="store_true",
                    help="round 2: more images of the names each class attracted in round 1 -> genpool_t.npz")
    ap.add_argument("--photo", action="store_true", help="with --targeted: plain photographic templates -> genpool_p.npz")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--per-prompt", type=int, default=3, help="images per prompt (different seeds)")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (tests)")
    args = ap.parse_args()
    if args.check:
        check(args)
    elif args.merge:
        merge(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
