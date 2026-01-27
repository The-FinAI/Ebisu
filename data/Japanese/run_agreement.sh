#!/usr/bin/env bash
set -euo pipefail

# ========= 可配置参数 =========

FILE="project-27-at-2026-01-22-15-03-5125843f.json"
MODE="choice"                 # choice 或 span_class (choice表示的是分类任务，比如你这边的intent任务，span_class是实体标注任务)
IGNORE_LABEL=true             # 只在 span_class 生效

# 留空 = 使用 Python 默认
FROM_NAME=""

# one-vs-others: true 表示算 A vs 所有人；false 表示算 A vs B（或默认两人）
ONE_VS_OTHERS=true

# A / B：只在需要时填写
ANN_A=24                      # 留空可设为 ""，则默认取文件里第一个 annotator （ha是24，so是22，zh是21.这个id信息我是从导出的json文件label studio结合着看才找到的，我目前只能这样找到对应的标注者编号）
ANN_B=""                      # ONE_VS_OTHERS=false 时可指定；否则留空

SCRIPT="compute_agreement_overall.py"   # 你的 python 脚本名

# ========= 构造 args（不要用 eval） =========

args=(--file "$FILE" --mode "$MODE")

# from_name
if [[ -n "${FROM_NAME}" ]]; then
  args+=(--from_name "$FROM_NAME")
fi

# span_class 时是否忽略 span label
if [[ "$MODE" == "span_class" && "${IGNORE_LABEL}" == "true" ]]; then
  args+=(--ignore_span_label)
fi

# one-vs-others
if [[ "${ONE_VS_OTHERS}" == "true" ]]; then
  args+=(--one_vs_others)
fi

# ann_a / ann_b
if [[ -n "${ANN_A}" ]]; then
  args+=(--ann_a "$ANN_A")
fi

if [[ "${ONE_VS_OTHERS}" != "true" && -n "${ANN_B}" ]]; then
  args+=(--ann_b "$ANN_B")
fi

# ========= 执行 =========

echo "Running agreement:"
echo "  SCRIPT      = $SCRIPT"
echo "  FILE        = $FILE"
echo "  MODE        = $MODE"
echo "  FROM_NAME   = ${FROM_NAME:-<default>}"
echo "  IGNORE      = $IGNORE_LABEL (only span_class)"
echo "  ONE_VS_OTHERS = $ONE_VS_OTHERS"
echo "  ANN_A       = ${ANN_A:-<auto>}"
if [[ "${ONE_VS_OTHERS}" != "true" ]]; then
  echo "  ANN_B       = ${ANN_B:-<auto/required if >2 annotators>}"
fi
echo

python "$SCRIPT" "${args[@]}"
