# Command line

Installing Dromeus provides the `dromeus` command (also `python -m dromeus`).
Each command makes one library call: anything the command does, you can also do
from Python.

| Command | Does | Library call |
| --- | --- | --- |
| `dromeus draft` | Writes the group's shared draft | `make_draft` |
| `dromeus check` | Checks a draft against your task and this machine, offline | `check_draft` |
| `dromeus doctor` | Checks that this machine is ready to run a node | — |
| `dromeus run` | Joins the group and trains every round | `Node.run` |
| `dromeus watch` | Follows a node's rounds, loss and consensus distance | — |

## Pointing at your task

`draft`, `check`, `run` and `doctor` take `--task module:callable`. Dromeus
imports `module` from the current directory, as `python -m` would, and calls
`callable()` with no arguments. It may return:

- the `TrainingTask`, enough for `draft`, `check` and `doctor`;
- `(TrainingTask, LocalData)`, which `run` needs, since a running node trains on
  its own records.

The [example](../examples/custom_training.py) has both: `task()` and `setup()`.

## `dromeus draft`

```bash
dromeus draft draft.yaml --task my_training:task --run-id regression-2026-11 \
  --participants 4 --rounds 20 --local-steps 10 --axl-version your-axl-version
```

Prints the draft hash. Uses the default NoLoCo and transport settings, without
privacy or divergence warnings. For those, call `dromeus.make_draft` from Python
(see [Write the shared draft](application-training.md#write-the-shared-draft)).

## `dromeus check`

```bash
dromeus check --draft draft.yaml --task my_training:task
```

Confirms that the model, task and loss definitions in the draft match your task
and that this machine has the Dromeus and PyTorch versions the draft names.
Prints the draft hash, so nodes can compare it. Doesn't touch the network.

## `dromeus doctor`

```bash
dromeus doctor --config node.yaml [--task my_training:task]
```

Runs these checks and prints one line for each, with a fix for anything not OK:

| Check | Fails when |
| --- | --- |
| `config` | The node configuration is invalid (no other check runs) |
| `draft` | The draft can't be read, or names other Dromeus or PyTorch versions |
| `task` | With `--task`: the draft was made from a different task |
| `axl bridge` | Nothing answers at `axl_bridge_url` |
| `node key` | The AXL bridge doesn't report this node's public key |
| `peers` | No AXL peer link is up |
| `invitation` | Participants only: the invitation is invalid, for another draft or bootstrap URI, or expired. Not delivered yet is a warning. |
| `clock` | The system clock isn't NTP-synchronized. On Linux this is read from `timedatectl`; elsewhere it's a warning that asks you to confirm NTP is on. |

## `dromeus run`

```bash
dromeus run --config node.yaml --task my_training:setup
```

Prints the run ID, committed rounds and manifest hash when the run completes.
Progress goes to the node's event log; follow it with `dromeus watch`.

## `dromeus watch`

```bash
dromeus watch --run-root runs/regression/node-0
```

Prints one line per event as the node writes it: formation, each committed round
(loss, public evaluation metrics, round time, peer) and each consensus distance.
It waits for the log to appear and stops when the run completes or fails. `--once`
prints what is logged so far and exits.

## Errors and exit codes

Errors print as one line on stderr. Add `--debug` for the full traceback.

| Code | Meaning |
| --- | --- |
| 0 | Success |
| 1 | Unexpected error: a bug, or an exception from your task code |
| 2 | Malformed command line |
| 3 | Invalid input: node configuration, draft, task or privacy settings (`ConfigError`, `DraftError`, `TaskError`, `PrivacyError`) |
| 4 | The group didn't form (`FormationError`) |
| 5 | The run failed after formation (`RunError`); also `watch` when the run failed |
| 6 | `doctor` found a failing check |
| 130 | Interrupted with Ctrl-C |
