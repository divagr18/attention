# Runpod setup

Use a CUDA template with at least 24 GB VRAM. An RTX 4090 is adequate for the
current one-layer 8K run; use 48 GB if adding layers, dense baselines, or a
larger batch. Do not expose a public SSH/Jupyter endpoint without an access
token.

## Pod workflow

1. Create a GPU pod with a persistent volume and clone/upload this repository
   onto that volume.
2. From the repository root, run `bash scripts/setup_runpod.sh`.
3. Run the 4K regression first:

   ```bash
   .venv-runpod/bin/python experiments/triton_kvcache_transformer_decode.py --checkpoint results/transformer_learned_page_staged_multirecord_4k.pt --output results/runpod_4k_regression.json
   ```

4. Launch the full 8K curriculum in `tmux` so it survives disconnects:

   ```bash
   tmux new -s cascade8k
   bash scripts/runpod_8k_curriculum.sh 2>&1 | tee results/runpod_8k.log
   ```

   If a run is already active without `tmux`, do not interrupt it. `tee` is
   still writing its output to `results/runpod_8k.log`; install `tmux` through
   the setup script before the next run.

5. Copy `results/*.json` and the final checkpoint back to persistent storage
   before stopping the pod. Stop the pod when idle; do not terminate it until
   artifacts are confirmed on the volume.

## Expected artifacts

- `transformer_learned_span{3,8,16,32}_8k.json`
- `transformer_learned_page_staged_8k.json`
- `triton_kvcache_page_staged_decode_8k.json`

The 8K launcher preserves the validated 4K training code and only changes the
context length. It deliberately trains page attention by progressive widening,
because direct full-page training failed at 4K.

For a fair 16K curriculum control, `scripts/runpod_16k_direct_page_control.sh`
starts from the same 3-token span checkpoint as the staged curriculum and
uses the identical 4,800 training updates under full-page exposure.
