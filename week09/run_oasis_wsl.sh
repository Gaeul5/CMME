#!/usr/bin/env bash
# 연구실 WSL에서 OASIS-3 새 피험자들을 FreeSurfer로 처리하고 week05/oasis_pipeline.py로 npz를 만든다.
# 원본 MRI와 FreeSurfer 결과는 WSL 밖(클라우드, GitHub)으로 옮기지 않는다. 가져올 것은 heads_oasis/*.npz 뿐.
#
# 사용법:
#   T1_DIR=~/oasis3_t1 bash week09/run_oasis_wsl.sh           # T1_DIR 안의 *.nii.gz 전부
#   T1_DIR=~/oasis3_t1 JOBS=2 bash week09/run_oasis_wsl.sh    # recon-all 2개 동시 (RAM 16GB 이상일 때만)
# 필요: FreeSurfer 7 (recon-all, mri_watershed), FREESURFER_HOME 설정, python3 + mne
set -euo pipefail
: "${FREESURFER_HOME:?FreeSurfer 환경을 먼저 불러오세요: source \$FREESURFER_HOME/SetUpFreeSurfer.sh}"
T1_DIR=${T1_DIR:?T1 파일이 있는 폴더를 T1_DIR로 지정}
export SUBJECTS_DIR=${SUBJECTS_DIR:-$HOME/oasis3_test/subjects_dir}
OUT=${OUT:-$HOME/oasis3_test/heads_oasis}
JOBS=${JOBS:-1}
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$SUBJECTS_DIR" "$OUT"
[ -d "$SUBJECTS_DIR/fsaverage" ] || cp -r "$FREESURFER_HOME/subjects/fsaverage" "$SUBJECTS_DIR/"

one() {
  t1=$1
  subj=$(basename "$t1" | grep -oE 'OAS3[0-9]{4}' | head -1)
  [ -n "$subj" ] || { echo "이름에 OAS 번호 없음: $t1"; return 0; }
  log="$SUBJECTS_DIR/$subj.log"
  if [ ! -f "$SUBJECTS_DIR/$subj/surf/lh.sphere.reg" ]; then
    echo "[$(date +%H:%M)] $subj recon-all 시작 (6~10시간)"
    rm -rf "$SUBJECTS_DIR/$subj"   # 중간에 끊긴 결과는 지우고 다시
    recon-all -s "$subj" -i "$t1" -all > "$log" 2>&1 || { echo "$subj recon-all 실패 → $log"; return 0; }
  fi
  if [ ! -f "$OUT/$subj.npz" ]; then
    seed=$((10#${subj#OAS3}))
    python3 "$REPO/week05/oasis_pipeline.py" --subjects-dir "$SUBJECTS_DIR" --subject "$subj" \
      --out "$OUT/$subj.npz" --seed "$seed" >> "$log" 2>&1 || { echo "$subj forward 실패 → $log"; return 0; }
  fi
  echo "[$(date +%H:%M)] $subj 끝"
}
export -f one
export SUBJECTS_DIR OUT REPO
ls "$T1_DIR"/*.nii* | xargs -P "$JOBS" -I{} bash -c 'one "$@"' _ {}
echo "완료: $(ls "$OUT"/*.npz 2>/dev/null | wc -l)개 → $OUT"
