#!/bin/bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
for DS in math aime asdiv gsm8k commonsenseqa openbookqa; do
    echo ""
    echo ">>>>>>>>>> CP-Router: ${DS} <<<<<<<<<<"
    bash "${DIR}/eval_cp_${DS}.sh"
done
echo "✅ All CP-Router done!"
