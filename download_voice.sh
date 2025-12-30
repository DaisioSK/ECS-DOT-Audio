#!/usr/bin/env bash
set -euo pipefail

URL="https://zenodo.org/records/3514950/files/VOICe_clean.7z?download=1"                     # 替换为实际下载地址
OUT_DIR="data/voice"
OUT_FILE="${OUT_DIR}/voice.zip"
MD5_EXPECTED="511838a9a2036ebfdc430dba85700a88"            # 如有校验值填上，否则留空

mkdir -p "${OUT_DIR}"

echo "Downloading to ${OUT_FILE} ..."
# 支持断点续传，自动重试
# curl -C - -L --retry 5 --retry-delay 3 --fail -o "${OUT_FILE}" "${URL}"

curl -L --fail --http1.1 \
  -C - \
  --retry 20 --retry-all-errors --retry-delay 5 \
  --speed-time 30 --speed-limit 10240 \
  --connect-timeout 30 --max-time 0 \
  -o "${OUT_FILE}" \
  "${URL}"

if [[ -n "${MD5_EXPECTED}" ]]; then
  echo "Verifying MD5..."
  MD5_ACTUAL=$(md5sum "${OUT_FILE}" | awk '{print $1}')
  if [[ "${MD5_ACTUAL}" != "${MD5_EXPECTED}" ]]; then
    echo "MD5 mismatch: expected ${MD5_EXPECTED}, got ${MD5_ACTUAL}"
    exit 1
  fi
fi








echo "Extracting..."
unzip -o "${OUT_FILE}" -d "${OUT_DIR}"

echo "Done. Location: ${OUT_DIR}"

