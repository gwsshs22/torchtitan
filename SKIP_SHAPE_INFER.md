# Stage Shape Recording & Inference Skip

This feature records pipeline stage input/output shapes during the first training run and uses
them on subsequent runs to bypass PyTorch's runtime `_shape_inference` — which otherwise runs
a full dummy forward pass (with inter-rank `send_object_list`/`recv_object_list`) on the first
`schedule.step()` call.

## What it does

1. **Recording mode** (`enable_stage_input_record`): During the first two training steps, hooks
   each pipeline stage's `forward()` to capture input AND output tensor shapes/dtypes/devices.
   Saves per-rank JSON files.

2. **Skip inference mode** (`enable_skip_shape_inference`): On subsequent runs, loads the
   recorded shapes and passes them as meta tensors to `PipelineStage(input_args=...,
   output_args=...)` at construction time, so `_shape_inference` is never triggered.

## How to use

### Step 1: Record stage shapes (first run)

```toml
[leto]
enable_stage_input_record = true
stage_inputs_folder = "stage_inputs"  # optional, default
```

Or via CLI: `--leto.enable_stage_input_record`

This will:
- Hook each `model_part.forward()` at step 1, capture inputs and outputs
- Save metadata to `{dump_folder}/stage_inputs/rank_{rank}.json` after step 1
- Automatically remove hooks

### Step 2: Skip shape inference (subsequent runs)

```toml
[leto]
enable_skip_shape_inference = true
stage_inputs_folder = "stage_inputs"  # must match recording folder
```

Or via CLI: `--leto.enable_skip_shape_inference`

This will:
- Load the recorded JSON at `PipelineStage` construction time
- Pass input/output shapes as meta tensors directly to `PipelineStage`
- PyTorch's `_shape_inference` (dummy forward + inter-rank comms) is skipped entirely
- Falls back to runtime inference gracefully if the file is missing or mismatched

## Example workflow

```bash
# Run 1: Record shapes during first iteration
python train.py --config config.toml --leto.enable_stage_input_record

# Creates: {dump_folder}/stage_inputs/rank_0.json, rank_1.json, etc.

# Run 2+: Skip shape inference using recorded shapes
python train.py --config config.toml --leto.enable_skip_shape_inference
```

## File format

```json
{
  "stages": [
    {
      "args": [{"shape": [1, 4096], "dtype": "torch.int64", "device": "cuda:0"}],
      "kwargs": {}
    },
    {
      "args": [{"shape": [1, 4096, 256], "dtype": "torch.bfloat16", "device": "cuda:0"}],
      "kwargs": {}
    }
  ],
  "stage_outputs": [
    {
      "args": [{"shape": [1, 4096, 256], "dtype": "torch.bfloat16", "device": "cuda:0"}],
      "kwargs": {}
    },
    {
      "args": [{"shape": [1, 4096, 256], "dtype": "torch.bfloat16", "device": "cuda:0"}],
      "kwargs": {}
    }
  ]
}
```

- `stages`: input shapes for each local stage (one entry per `model_part` on this rank)
- `stage_outputs`: output shapes for each local stage

## Notes

- Recording and skip are per-rank (each rank writes/reads its own JSON file)
- Works with looped (`Interleaved1F1B`) and V-style (`DualPipeV`, `ZBVZeroBubble`) schedules
- If the shape file is missing or has mismatched stage count, falls back silently to runtime
  shape inference — no crash
- Both flags are independent; you can record on every run without enabling skip
