# Data Type Tracking

This document tracks data types as they flow through the sl-suite2p pipelines.

## IO Module Output

| Data | dtype | Notes |
|------|-------|-------|
| Binary frame data | `int16` | Written to `channel_N_data.bin` files |
| Mean images | `float32` | Stored in `RuntimeContext.runtime.detection.mean_image` |
