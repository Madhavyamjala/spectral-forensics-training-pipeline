#!/usr/bin/env bash
# Full training run (all videos, >=24 GB GPU(s)): trains, evaluates, exports and offers to push to the Hub.
exec bash "$(dirname "$0")/launch.sh" configs/full.yaml "$@"
