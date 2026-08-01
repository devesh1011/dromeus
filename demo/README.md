# Containerized Dromeus demo

This mode runs four independent Dromeus workers in separate Docker containers.
Each worker owns one AXL process, one Dromeus runtime, a private identity, and a
separate artifact volume. The worker starts AXL before formation, connects Dromeus
to its local AXL HTTP bridge, and stops AXL when the run ends. The dashboard is a
fifth container and observes worker state over the Docker network.

```bash
docker compose -f demo/compose.yaml up --build -d
docker compose -f demo/compose.yaml ps
```

Open `http://localhost:8765` in a browser. The Compose file targets
`linux/amd64`; Docker Desktop runs it through emulation on Apple Silicon, so the
first build and training run may be slower.

Choose number D-PSGD rounds in **Training rounds**, then click **Begin formation**
to form fixed group. Click **Start training** to begin round 0. First training run
downloads CIFAR-10 into shared Docker volume; later runs reuse it. Dashboard's
**Live training logs** panel updates after each committed round.

After training finishes, the dashboard remains in `Training complete` and does not
replay the formation sequence.

The same two explicit stages can be controlled from a terminal:

```bash
uv run python -m demo.formation.cli form --rounds 5 --wait
# Nothing trains while status is "formed".
uv run python -m demo.formation.cli train --follow
```

For a deployed demo, pass its current dashboard URL:

```bash
uv run python -m demo.formation.cli \
  --url http://HOSTNAME_OR_IP:8765 state
```

To show the live container evidence during a presentation:

```bash
docker stats --no-stream
docker compose -f demo/compose.yaml logs -f node-0 node-1 node-2 node-3
```

Stop everything with:

```bash
docker compose -f demo/compose.yaml down -v
```

The `-v` removes demo-only identities and artifacts so the next run starts clean.
