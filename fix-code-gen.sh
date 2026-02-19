cd ~/ACL-ARR-Jan-Rebuttal/OThink-R1

# run_standard.sh
sed -i '/^CMD="conda run/s|CMD="conda run -n othink-r1 -- uv run python|CMD="uv run python|' benchmark/livecodebench/run_standard.sh

# run_deer.sh
sed -i '/^CMD="conda run/s|CMD="conda run -n othink-r1 -- uv run python|CMD="uv run python|' benchmark/livecodebench/run_deer.sh

echo "✅ 已修复 (去掉 conda run，直接用 uv run)"