#!/usr/bin/env bash
# Full two-class run: Real vs AI-Generated only.
# Runs every training/evaluation/export stage through the normal resumable driver.
exec bash "$(dirname "$0")/launch.sh" configs/full_2class.yaml "$@"
