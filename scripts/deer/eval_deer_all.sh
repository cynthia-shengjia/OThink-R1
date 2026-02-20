#!/bin/bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
for DS in math aime gsm8k asdiv commonsenseqa openbookqa; do
    echo ""
    echo ">>>>>>>>>> Deer: ${DS} <<<<<<<<<<"
    bash "${DIR}/eval_deer_${DS}.sh"
done
echo "✅ All Deer done!"
