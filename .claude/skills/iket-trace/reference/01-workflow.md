# Collecting a Trace

## Prerequisite check

```bash
run-iket --help
```

If this fails, the install predates 4.6.0 or is incomplete. `run-iket` is a console
script shipped with `nvidia-cutlass-dsl`; it wraps `iket.cli.main`. There is no
standalone `iket` package on PyPI, so the only way to obtain the collector is a
sufficiently new `nvidia-cutlass-dsl`.

The instrumentation API (`cutlass.cute.experimental.iket`) can be present while the
collector is absent. In that state you can annotate but collect nothing, and the failure
is silent.

## Command

```bash
run-iket --output-dir ./iket_out --clobber \
  profile --postprocess all -- \
  python your_workload.py --your --args
```

The workload command must appear after `--`, otherwise its arguments are parsed as
profiler arguments.

### Options

| Option | Meaning |
|---|---|
| `--output-dir DIR`, `-o DIR` | Where traces and intermediates go |
| `--clobber` | Replace an existing output directory without prompting |
| `--working-dir DIR` | Working directory for the profiled process |
| `--log-level {error,warn,info,debug,trace}` | Default `error`; raise it when a run produces nothing |
| `--context-buffer-size N` | Extra device buffer per context; default auto-computed, fallback 1G |
| `--use-injection-lib PATH` | Override the bundled injection library |
| `--skip-run` | Skip the real run |

Subcommands: `profile` (collect) and `postprocess` (regenerate outputs from an existing
run directory, so a large collection need not be repeated to change export format).

`--postprocess` accepts `perfetto`, `json`, or `all`.

## Two-pass model

`run-iket` runs the workload **twice**: a dry pass that estimates how much device buffer
the workload needs, then the real pass that allocates with margin and collects records.
Budget wall time accordingly, and make sure the workload is deterministic enough that the
second pass emits a similar record count. If per-warp record counts vary wildly between
passes, buffers may be undersized — raise `--context-buffer-size`.

## Artifacts

```
iket_out/
  iket_pid_0x<pid>.pftrace        Perfetto timeline
  iket_pid_0x<pid>.trace.json     machine-readable
  iket_pid_0x<pid>.pftrace.gz     compressed timeline
  iket_pid_0x<pid>.html           self-contained viewer
```

Intermediate directories (`tracker/`, `gen-config/`, `iket/`) are removed on success.

## Viewing

Serve the directory and open the generated HTML:

```bash
cd iket_out && python3 -m http.server 8080
```

Or import the `.pftrace` at <https://ui.perfetto.dev/>. Perfetto is only a viewer; the
trace content comes entirely from the kernel instrumentation.

For a headless machine, skip the UI and run `helpers/analyze_trace.py` on the JSON.

## Instrumentation without run-iket

To build instrumented kernels outside a profiling run:

```bash
export CUTE_DSL_COMPILER_OPT=iket
```

or per compilation:

```python
compiled = cute.compile(host_function, *args, options="iket")
```

This only enables lowering. It does not collect anything, so it is useful for inspecting
generated code, not for getting traces.
