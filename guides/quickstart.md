# Quickstart: four nodes on one machine

Train the [example regression task](../examples/custom_training.py) with four
Dromeus nodes, each beside its own AXL node, on one computer. It takes a few
minutes and needs no network setup. A test runs these exact commands, so they
stay correct.

You need Linux (x86_64 or aarch64) or macOS on Apple silicon, Git, and
[uv](https://docs.astral.sh/uv/getting-started/installation/). On these
platforms Dromeus installs a pinned AXL build, so you don't need Go.

## 1. Install

```bash
git clone https://github.com/devesh1011/dromeus.git
cd dromeus
./scripts/bootstrap
```

Run the remaining steps from the `dromeus` directory, in one terminal.

## 2. Write the data and the shared draft

```bash
source .venv/bin/activate
python examples/quickstart/make_data.py runs/quickstart
dromeus draft runs/quickstart/draft.yaml --task examples.custom_training:task \
  --run-id quickstart --participants 4 --rounds 20 --local-steps 10
```

Each node gets its own records (`data-0.npz` ... `data-3.npz`). The draft is the
group's agreement: model, task, loss, optimizer, rounds, and the Dromeus,
PyTorch and AXL versions. In a real group one person writes it and sends the file
to the others.

## 3. Start four AXL nodes

```bash
for i in 0 1 2 3; do
  dromeus-axl -config examples/quickstart/axl-$i.json > runs/quickstart/axl-$i.log 2>&1 &
done
sleep 5
dromeus doctor --config examples/quickstart/node-0.yaml
```

[`axl-0.json`](../examples/quickstart/axl-0.json) listens for peers on
`127.0.0.1:9101`; the other three connect out to it, as nodes behind a NAT would.
Each node's HTTP bridge is on its own port, 9201 to 9204. `dromeus doctor` should
report `ok` for everything except, perhaps, the clock, and `connected to 3
peer(s)` for node 0.

## 4. Train

```bash
nodes=()
for i in 0 1 2 3; do
  DROMEUS_LOCAL_DATA=runs/quickstart/data-$i.npz \
    dromeus run --config examples/quickstart/node-$i.yaml \
    --task examples.custom_training:setup > runs/quickstart/node-$i.log 2>&1 &
  nodes+=($!)
done
dromeus watch --run-root runs/quickstart/node-0
wait "${nodes[@]}"
tail -n 1 runs/quickstart/node-*.log
```

Node 0 is the initiator: it writes `runs/quickstart/invitation.json`, which the
participants read from the same path (on separate machines you'd copy the file).
`dromeus watch` shows node 0's rounds: loss on its own records, error on a public
holdout, and the distance between nodes' models. Every node ends with `run
quickstart complete: 20 rounds committed` and the same manifest hash.

## 5. Stop the AXL nodes

```bash
pkill -f 'examples/quickstart/axl-'
```

To run again, start over from step 2 after `rm -rf runs/quickstart`: a run's
state stays in its `run_root`.

## Next

- [AXL setup](axl.md): run AXL across machines, with a stable identity, a public
  bootstrap peer and nodes behind NAT.
- [Training with Dromeus](application-training.md): write your own task.
- [Configuration](configuration.md): the node configuration and the draft.
