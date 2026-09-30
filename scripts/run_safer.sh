#!/usr/bin/env bash
# SAFER, the paper's configuration: Real vs AI-Generated on Chrono-66k (configs/safer.yaml).
# Runs every training / evaluation / export stage through the normal resumable driver.
exec bash "$(dirname "$0")/launch.sh" configs/safer.yaml "$@"
