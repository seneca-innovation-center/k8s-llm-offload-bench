#!/bin/bash
# Download the top-level files of a Hugging Face model repo with curl (resumable), verify sizes. Usage: dl_model.sh REPO DIR
REPO=$1; DIR=$2; mkdir -p "$DIR"; cd "$DIR" || exit 1
BASE=https://huggingface.co/$REPO/resolve/main
curl -s "https://huggingface.co/api/models/$REPO/tree/main" > tree.json
FILES=$(python3 -c "
import json; t=json.load(open('tree.json'))
print(' '.join(f['path'] for f in t if f['type']=='file' and '/' not in f['path'] and f['path'].rsplit('.',1)[-1] in ('json','safetensors','txt','jinja','model')))")
echo "$REPO: $FILES"
for f in $FILES; do curl -sL -C - --retry 30 --retry-delay 5 --retry-all-errors -o "$f" "$BASE/$f" & done; wait
python3 - <<PY
import json, os
t = json.load(open("tree.json")); ok = True
for f in t:
    p = f["path"]
    if f["type"] != "file" or "/" in p or p.rsplit(".", 1)[-1] not in ("json", "safetensors", "txt", "jinja", "model"): continue
    want = (f.get("lfs") or {}).get("size", f.get("size")); have = os.path.getsize(p) if os.path.exists(p) else -1
    ok &= have == want
    if have != want: print("MISMATCH", p, have, want)
print("$REPO", "OK" if ok else "SIZE MISMATCH")
PY
