#!/bin/bash
# 自动化运行LLM-backbone实验脚本
# 仅运行beijing，依次下载并测试多个模型

set -euo pipefail

BASE_DIR="/workspace/China_Journal"
BASE_CONFIG="${BASE_DIR}/config/config.yaml"
DATADISK="/datadisk"
LOG_DIR="${BASE_DIR}/output/logs"

# 需要下载和测试的模型列表（按顺序执行）
# 可选第三、四字段：skip_download、skip_rag，用于复用已有模型/已有RAG
MODELS=(
  "Llama-2-7B|LLM-Research/llama-2-7b|skip_download|skip_rag"  # 已有模型与RAG，跳过下载和构建
  "Llama-2-13B|shakechen/Llama-2-13b"
  "Qwen2.5-1.5B-Instruct|Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen2.5-7B-Instruct|Qwen/Qwen2.5-7B-Instruct"
)

mkdir -p "$DATADISK" "$LOG_DIR"

for ITEM in "${MODELS[@]}"; do
  IFS='|' read -r MODEL_NAME MODEL_REPO FLAG1 FLAG2 <<< "$ITEM"
  MODEL_DIR="${DATADISK}/${MODEL_NAME}"

  SKIP_DOWNLOAD=false
  SKIP_RAG=false
  for FLAG in "$FLAG1" "$FLAG2"; do
    case "$FLAG" in
      skip_download) SKIP_DOWNLOAD=true ;;
      skip_rag) SKIP_RAG=true ;;
    esac
  done

  if [ "$SKIP_DOWNLOAD" = true ] && [ ! -d "$MODEL_DIR" ]; then
    echo "[WARN] ${MODEL_NAME} 标记为跳过下载，但目录不存在，仍将下载。"
    SKIP_DOWNLOAD=false
  fi

  if [ "$SKIP_DOWNLOAD" = true ]; then
    echo "==== 跳过下载 ${MODEL_NAME}（使用已有目录 ${MODEL_DIR}） ===="
  else
    echo "==== 下载模型 ${MODEL_NAME} 到 ${MODEL_DIR} ===="
    rm -rf "$MODEL_DIR"
    modelscope download --model "$MODEL_REPO" --local_dir "$MODEL_DIR"
  fi

  echo "==== 为 ${MODEL_NAME} 生成专用配置 ===="
  TEMP_CONFIG="$(mktemp)"
  python - "$BASE_CONFIG" "$TEMP_CONFIG" "$MODEL_NAME" "$MODEL_DIR" <<'PY'
import sys
import yaml

base_config, temp_config, model_name, model_path = sys.argv[1:5]
with open(base_config, 'r') as f:
    cfg = yaml.safe_load(f)

cfg['data']['city'] = 'beijing'
cfg['model']['llm']['model_name'] = model_name
cfg['model']['llm']['model_path'] = model_path

with open(temp_config, 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

  if [ "$SKIP_RAG" = true ]; then
    echo "==== 跳过构建 ${MODEL_NAME} 的 RAG（已存在） ===="
  else
    echo "==== 构建 ${MODEL_NAME} 的 RAG 数据库（beijing） ===="
    python "${BASE_DIR}/util/build_rag_database.py" \
      --config "$TEMP_CONFIG" \
      --city beijing \
      --model_name "$MODEL_NAME"
  fi

  echo "==== 运行测试并记录日志 ===="
  LOG_FILE="${LOG_DIR}/test_${MODEL_NAME}.log"
  bash "${BASE_DIR}/scripts/test.sh" "$TEMP_CONFIG" > "$LOG_FILE" 2>&1

  echo "==== ${MODEL_NAME} 测试完成，日志: ${LOG_FILE} ===="

  rm -f "$TEMP_CONFIG"
done

echo "所有模型测试完成。"
