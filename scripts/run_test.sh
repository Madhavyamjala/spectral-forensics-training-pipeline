#!/usr/bin/env bash
# Test run (5 000 videos, 12-16 GB GPU): checks the full training pipeline runs without errors.
exec bash "$(dirname "$0")/launch.sh" configs/test.yaml "$@"
