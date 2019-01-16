#!/bin/bash

SCRIPT_DIRECTORY="$( dirname "${BASH_SOURCE[0]}" )"

set -e

for script in $SCRIPT_DIRECTORY/*.py; do
    echo + $script
    python $script
done
