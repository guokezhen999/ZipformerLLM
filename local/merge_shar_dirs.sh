#!/bin/bash
# 将多个 Shar 目录通过软链接合并到一个目录下。
# 每个源目录可指定一个唯一 prefix，自动重命名为 cuts.{prefix}_{编号}.jsonl.gz 避免冲突。
#
# 用法:
#   bash local/merge_shar_dirs.sh \
#     --target-dir data/fbank/merged_shar/train \
#     --source-dirs "data/fbank/shar/a,data/fbank/shar/b" \
#     --prefixes "a,b"
#
# 命名规则:
#   cuts.{prefix}_{shard_num}.jsonl.gz
#   features.{prefix}_{shard_num}.tar

set -euo pipefail

target_dir=""
source_dirs_arg=""
prefixes_arg=""

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

usage() {
  cat <<EOF
Usage:
  bash local/merge_shar_dirs.sh \\
    --target-dir <merged_dir> \\
    --source-dirs <dir1,dir2,...> \\
    --prefixes <prefix1,prefix2,...>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-dir)
      target_dir="$2"; shift 2 ;;
    --source-dirs)
      source_dirs_arg="$2"; shift 2 ;;
    --prefixes)
      prefixes_arg="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      log "ERROR: unknown arg: $1"
      usage
      exit 1 ;;
  esac
done

if [[ -z "$target_dir" || -z "$source_dirs_arg" || -z "$prefixes_arg" ]]; then
  log "ERROR: --target-dir, --source-dirs, --prefixes are required"
  usage
  exit 1
fi

IFS=',' read -r -a source_dirs <<< "$source_dirs_arg"
IFS=',' read -r -a prefixes <<< "$prefixes_arg"

# trim spaces
for i in "${!source_dirs[@]}"; do
  source_dirs[$i]="$(echo "${source_dirs[$i]}" | xargs)"
done
for i in "${!prefixes[@]}"; do
  prefixes[$i]="$(echo "${prefixes[$i]}" | xargs)"
done

if [ ${#source_dirs[@]} -ne ${#prefixes[@]} ]; then
  log "ERROR: source_dirs (${#source_dirs[@]}) and prefixes (${#prefixes[@]}) length mismatch!"
  exit 1
fi

n_sources=${#source_dirs[@]}

valid_sources=()
for ((i = 0; i < n_sources; i++)); do
  d="${source_dirs[$i]}"
  p="${prefixes[$i]}"
  if [ -d "$d" ]; then
    n=$(find "$d" -maxdepth 1 -name "cuts.*.jsonl.gz" 2>/dev/null | wc -l)
    if [ "$n" -gt 0 ]; then
      log "Found [$p]: $d  ($n shards)"
      valid_sources+=("$i")
    else
      log "WARNING [$p]: $d exists but has no cuts.*.jsonl.gz, skipping"
    fi
  else
    log "WARNING [$p]: $d does not exist, skipping"
  fi
done

if [ ${#valid_sources[@]} -eq 0 ]; then
  log "ERROR: No valid source directories found."
  exit 1
fi

if [ -d "$target_dir" ]; then
  bak="${target_dir}_bak_$(date +%Y%m%d_%H%M%S)"
  log "Target dir exists, moving to: $bak"
  mv "$target_dir" "$bak"
fi
mkdir -p "$target_dir"

total_shards=0
total_files=0

for idx in "${valid_sources[@]}"; do
  src="${source_dirs[$idx]}"
  prefix="${prefixes[$idx]}"

  log "Linking [$prefix]: $src"

  mapfile -t cuts_files < <(find "$src" -maxdepth 1 -name "cuts.*.jsonl.gz" | sort -t. -k2 -n)

  original_num="000000"
  for cuts_file in "${cuts_files[@]}"; do
    base=$(basename "$cuts_file")
    original_num=$(echo "$base" | sed 's/cuts\.\([0-9]*\)\.jsonl\.gz/\1/')
    dir=$(dirname "$cuts_file")
    feats_file="$dir/features.$original_num.tar"

    new_cuts="$target_dir/cuts.${prefix}_${original_num}.jsonl.gz"
    new_feats="$target_dir/features.${prefix}_${original_num}.tar"

    ln -sf "$(realpath "$cuts_file")" "$new_cuts"
    ((total_files++)) || true

    if [ -f "$feats_file" ]; then
      ln -sf "$(realpath "$feats_file")" "$new_feats"
      ((total_files++)) || true
    else
      log "  WARNING: missing features file: $(basename "$feats_file")"
    fi
  done

  n_shards=${#cuts_files[@]}
  log "  $n_shards shards linked (${prefix}_000000 ~ ${prefix}_$(printf "%06d" $((10#$original_num))))"
  ((total_shards += n_shards)) || true
done

log "=========================================="
log "Done. $total_shards shards, $total_files files → $target_dir"
log "=========================================="

echo ""
echo "Directory listing (first 10):"
ls -1 "$target_dir" | head -10
echo "  ..."
echo ""
echo "Training:"
echo "  --train_shar_dirs $target_dir"
echo ""
