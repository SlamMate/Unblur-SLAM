#!/bin/bash
echo "=== Tmux 会话中的 Slurm 作业 ==="
tmux list-panes -a -F "#{session_name}:#{window_index}.#{pane_index}" | while read pane; do
    job_id=$(tmux display-message -p -t "$pane" '#{pane_pid}' | xargs -I {} ps -o command= -p {} | grep -oP 'SLURM_JOB_ID=\K\d+' || echo "")
    if [ -n "$job_id" ]; then
        echo "Pane: $pane -> Job ID: $job_id"
    fi
done
