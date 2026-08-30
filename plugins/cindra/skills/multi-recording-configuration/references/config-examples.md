# Multi-recording configuration examples

Shows the minimal, typical, and full multi-recording configuration files, along with the top-level `pipeline_type`
discriminator every one of them carries.

---

## Minimal configuration (required fields only)

```yaml
pipeline_type: multi-recording

recording_io:
  dataset_name: "animal_A_learning_task"
```

The `pipeline_type: multi-recording` discriminator must be present at the top level of every multi-recording
configuration file. `generate_config_file_tool` writes it automatically, and both `validate_config_file_tool` and
`cindra run` reject a file that omits it. The examples below show only the sections being customized and assume the
discriminator is already present.

---

## Typical configuration

```yaml
recording_io:
  dataset_name: "animal_A_learning_task"

roi_selection:
  probability_threshold: 0.85
  maximum_size: 1000

diffeomorphic_registration:
  speed_factor: 3.0

roi_tracking:
  threshold: 0.75
  mask_prevalence: 50
```

---

## Full configuration with MROI region filtering

```yaml
runtime:
  display_progress_bars: false

recording_io:
  dataset_name: "animal_A_vr_navigation"

roi_selection:
  probability_threshold: 0.85
  maximum_size: 1000
  mroi_region_margin: 30

diffeomorphic_registration:
  image_type: "enhanced_mean"
  grid_sampling_factor: 1.0
  final_grid_sampling: 16.0
  scale_sampling: 30
  speed_factor: 3.0

roi_tracking:
  threshold: 0.75
  mask_prevalence: 50
  pixel_prevalence: 50
  step_sizes: [200, 200]
  bin_size: 50
  maximum_distance: 20
  minimum_size: 25
```

The `signal_extraction` and `spike_deconvolution` sections take the same keys and defaults their parameter tables
state, and both are left at those defaults in the common case.
