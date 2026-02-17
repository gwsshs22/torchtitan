# Stage Warmup Feature

This feature allows you to record and replay inputs to pipeline parallel stages for warmup purposes.

## What it does

1. **Recording Mode**: During the first training iteration, records the shapes, dtypes, and devices of inputs to each pipeline stage
2. **Warmup Mode**: Before training starts (after checkpoint loading), replays the recorded inputs with synthetic data to warmup torch.compile

This is particularly useful for warming up both forward and backward compilation graphs without needing actual data.

## How to use

### Step 1: Record stage inputs

Run training with recording enabled:

```bash
# In your training config TOML file, add:
[leto]
enable_stage_input_record = true
stage_inputs_folder = "stage_inputs"  # optional, default is "stage_inputs"

# Or via command line:
--leto.enable_stage_input_record
```

This will:
- Hook into each model_part's forward method during first iteration
- Save recorded metadata to `{dump_folder}/stage_inputs/rank_{rank}.json`
- Automatically remove hooks after first iteration

### Step 2: Warmup with recorded inputs

In subsequent runs, enable warmup:

```bash
# In your training config TOML file:
[leto]
enable_stage_warmup = true
stage_inputs_folder = "stage_inputs"  # must match recording folder

# Or via command line:
--leto.enable_stage_warmup
```

This will:
- Load recorded metadata from `{dump_folder}/stage_inputs/rank_{rank}.json`
- Create synthetic tensors matching recorded shapes/dtypes
- Run forward pass on each stage (with gradients enabled for compile warmup)
- Clean up autograd graphs and free memory

## Example workflow

```bash
# Run 1: Record inputs during first iteration
python train.py --config config.toml --leto.enable_stage_input_record

# This creates: outputs/stage_inputs/rank_0.json, rank_1.json, etc.

# Run 2+: Use recorded inputs for warmup
python train.py --config config.toml --leto.enable_stage_warmup
```

## How it works internally

### Recording
1. Before first iteration: `StageInputRecorder` hooks `model_part.forward` methods
2. During first iteration: Each hook captures args/kwargs shapes and dtypes
3. After first iteration: Saves metadata to JSON and removes hooks

### Warmup
1. After checkpoint loading: Loads metadata from JSON
2. For each model_part:
   - Creates synthetic `torch.randn()` tensors with recorded shapes/dtypes
   - Calls `model_part(*synthetic_args, **synthetic_kwargs)` with `requires_grad=True`
   - This triggers torch.compile to create backward graph (AOT Autograd)
3. Cleanup: Deletes tensors, which automatically releases autograd graph
4. Final: `gc.collect()` and `torch.cuda.empty_cache()`

## File format

The recorded JSON file has this structure:

```json
{
  "stages": [
    {
      "args": [
        {"shape": [2, 1024], "dtype": "torch.float32", "device": "cuda:0"}
      ],
      "kwargs": {
        "attention_masks": {"shape": [2, 1024], "dtype": "torch.bool", "device": "cuda:0"}
      }
    },
    ...
  ]
}
```

## Notes

- Recording and warmup are per-rank (each rank has its own JSON file)
- Works with both single-stage and multi-stage pipeline schedules
- Each model_part (virtual stage) is recorded and warmed up independently
- No backward pass is run during warmup - cleanup happens by deleting tensors
- Compatible with torch.compile - the warmup triggers AOT Autograd compilation

## Debugging

Enable debug logging to see detailed information:

```bash
export LOG_LEVEL=DEBUG
python train.py --config config.toml --leto.enable_stage_warmup
```

You'll see logs like:
```
[INFO] Installed recording hooks on 2 model_parts
[DEBUG] Recorded inputs for model_part 0: 1 args, 1 kwargs
[INFO] Saved stage inputs to outputs/stage_inputs/rank_0.json
[INFO] Starting stage warmup for 2 model_parts...
[DEBUG] Warming up stage 0 with 1 args, 1 kwargs
[INFO] Stage warmup completed successfully
```
