# Containerized formation demo

This mode runs four independent Dromeus workers in separate Docker containers.
Each worker owns one AXL process, one Dromeus runtime, a private identity, and a
separate artifact volume. The dashboard is a fifth container and observes worker
state over the Docker network.

```bash
docker compose -f demo/compose.yaml up --build -d
docker compose -f demo/compose.yaml ps
open http://localhost:8765
```

Choose number D-PSGD rounds in **Training rounds**, then click **Begin formation**
to form fixed group. Click **Start training** to begin round 0. First training run
downloads CIFAR-10 into shared Docker volume; later runs reuse it. Dashboard's
**Live training logs** panel updates after each committed round.

The same two explicit stages can be controlled from a terminal:

```bash
uv run python -m demo.formation.cli form --rounds 5 --wait
# Nothing trains while status is "formed".
uv run python -m demo.formation.cli train --follow
```

Against the AWS demo, pass its dashboard URL:

```bash
uv run python -m demo.formation.cli \
  --url http://3.236.134.228:8765 state
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
