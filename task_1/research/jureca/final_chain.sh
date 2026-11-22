#!/bin/bash
# One command per final submission (run on the login node from the workspace):
#   NAME=sub14 FAMILY="tf tg" W=1 research/jureca/final_chain.sh
# builds the ensemble logits (sub10 members + each FAMILY seed repeated W times), prints the OOF, then submits the
# SAM refine + rectangle bet + assembly job (devel partition). Every output file is pre-created (inode quota).
set -euo pipefail
cd ${TEAM_DIR:?set TEAM_DIR}/wmloc
export PYTHONDONTWRITEBYTECODE=1
: "${NAME:?}"; : "${FAMILY:?}"; W=${W:-1}
L=outputs/logits_$NAME; O=outputs/sam_$NAME
mkdir -p $L $O
for m in pixelseal trustmark stegastamp; do
  touch $L/$m.npz $O/$m.npz $O/${m}_bet.npz $O/refine-$m.log $O/bet-$m.log
done
touch $O/assemble.log outputs/submissions/$NAME.pt research/submissions/$NAME.json outputs/jureca/wmloc-final-$NAME.out
r(){ local p=$1 s=$2; local o=""; for x in $p; do for k in 0 1 2 3 4; do o="$o,$x${k}_$s"; done; done; echo "$o"; }
rep(){ local s=$1; local o=""; for i in $(seq 1 $W); do o="$o$(r "$FAMILY" $s)"; done; echo "$o"; }
PSB="h1_wm_s1,h1_wm,j_wm_s1,f1_wm,k2_wm,k3_wm,k4_wm"
TMB="h1_r50,v2_r50,j_wm,f1_wm,f1_r50,f2cut_wm,k2_resnet50,k3_resnet50,k4_resnet50,k2_wm,k3_wm,k4_wm"
SSB="h1_r50,v2_r34,f1_r50,k2_resnet50,k3_resnet50,k4_resnet50"
V="cm cs cp cx rp"
.venv/bin/python -m research.build_sub_v2 --name $NAME --dry-run --save-logits $L \
  --rect pixelseal=outputs/rect/pixelseal.json:0.85:0.5 --rect trustmark=outputs/rect/trustmark.json:0.99:0.5 \
  --members pixelseal=$PSB$(r "$V" wm)$(rep wm) \
  --members trustmark=$TMB$(r "$V" wm)$(r "$V" resnet50)$(rep wm)$(rep resnet50) \
  --members stegastamp=$SSB$(r "$V" resnet50)$(rep resnet50) 2>&1 | grep -E "OOF|members not"
[[ ${DRY:-} ]] && exit 0
research/jureca/safe_sbatch.sh general research/jureca/final_chain.sbatch --export=ALL,LOGITS=$L,OUT=$O,NAME=$NAME,NOTE="FAMILY=$FAMILY W=$W" \
  --output=outputs/jureca/wmloc-final-$NAME.out --open-mode=truncate
