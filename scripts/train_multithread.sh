#!/bin/bash

uv run torchrun \
  --nnodes=1 \
  --nproc_per_node=5 \
  --master-addr=127.0.0.1 \
  --master-port=29500 \
  --local-addr=127.0.0.1 \
  -m transpressor.transpressor
