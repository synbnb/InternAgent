#!/bin/bash
# Run a sci_tasks task
# Usage:
#   bash scripts/run_sci.sh                                   # default: Astronomy_000
#   bash scripts/run_sci.sh <task_name>                       # specify task name
#   bash scripts/run_sci.sh sci_tasks/tasks/<task_name>       # specify full path

TASK="${1:-Astronomy_000}"

# If a bare name (no slash), prepend sci_tasks/tasks/
if [[ "$TASK" != */* ]]; then
    TASK="sci_tasks/tasks/$TASK"
fi

echo "=== Running sci_task: $TASK ==="

python launch_discovery.py \
    --config config/default_config.yaml \
    --task "$TASK" \
    --exp_backend claudecode
