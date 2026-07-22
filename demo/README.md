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

Click **Begin formation** in the dashboard. To show the live container evidence
during a presentation:

```bash
docker stats --no-stream
docker compose -f demo/compose.yaml logs -f node-0 node-1 node-2 node-3
```

Stop everything with:

```bash
docker compose -f demo/compose.yaml down -v
```

The `-v` removes demo-only identities and artifacts so the next run starts clean.
