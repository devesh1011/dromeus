const ui = {
  body: document.body,
  start: document.querySelector("#start-run"),
  startTraining: document.querySelector("#start-training"),
  startLabel: document.querySelector(".button-ready"),
  startBusyLabel: document.querySelector(".button-busy"),
  elapsed: document.querySelector("#elapsed"),
  status: document.querySelector("#run-status"),
  statusDetail: document.querySelector("#status-detail"),
  runId: document.querySelector("#run-id"),
  manifest: document.querySelector("#manifest-hash"),
  transportCopy: document.querySelector("#transport-copy"),
  agreement: document.querySelector("#agreement"),
  agreementCopy: document.querySelector("#agreement-copy"),
  phaseCount: document.querySelector("#phase-count"),
  phaseItems: [...document.querySelectorAll("#phase-list li")],
  ledger: document.querySelector("#event-ledger"),
  ledgerEmpty: document.querySelector("#ledger-empty"),
  eventCount: document.querySelector("#event-count"),
  topology: document.querySelector("#topology"),
  routes: document.querySelector("#routes"),
  packetLayer: document.querySelector("#packet-layer"),
  errorDialog: document.querySelector("#error-dialog"),
  errorMessage: document.querySelector("#error-message"),
  roundCount: document.querySelector("#round-count"),
  trainingProgress: document.querySelector("#training-progress"),
  trainingLog: document.querySelector("#training-log"),
};

const stateLabels = {
  offline: "Offline",
  booting: "Booting AXL",
  connected: "AXL connected",
  invited: "Invited",
  enrolling: "Enrolling peers",
  joining: "Join sent",
  accepted: "Accepted",
  sealing: "Sealing manifest",
  manifest: "Manifest received",
  syncing: "Checkpoint sync",
  verifying: "Verifying",
  waiting: "Ready barrier",
  starting: "Finalizing formation",
  training: "Training",
  ready: "Formation ready",
  complete: "Training complete",
  failed: "Failed",
};

const eventLabels = {
  INVITATION: "INVITE / OOB",
  JOIN_REQUEST: "JOIN REQUEST",
  JOIN_ACCEPTED: "JOIN ACCEPTED",
  MANIFEST_SEALED: "SEALED MANIFEST",
  TRANSFER_BEGIN: "CHECKPOINT",
  CHUNK: "MODEL PAYLOAD",
  CHUNK_ACK: "PAYLOAD ACK",
  TRANSFER_COMPLETE: "VERIFY",
  READY: "READY",
  START: "FORMATION START",
  START_ACK: "FORMATION ACK",
};

const phaseRules = [
  { type: "INVITATION", key: "target", required: 1 },
  { type: "JOIN_ACCEPTED", key: "target", required: 3 },
  { type: "MANIFEST_SEALED", key: "target", required: 3 },
  { type: "TRANSFER_COMPLETE", key: "target", required: 3 },
  { type: "READY", key: "source", required: 3 },
  { type: "START_ACK", key: "source", required: 3 },
];

const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
let backend = null;
let generation = null;
let seenSequences = new Set();
let processedEvents = [];
let phaseEvidence = phaseRules.map(() => new Set());
let visualEpoch = 0;
let shownErrorGeneration = null;
let requestInFlight = null;
let pendingRequest = null;
let clientError = null;

function nodeElement(nodeId) {
  return document.getElementById(nodeId);
}

function setNode(nodeId, nextState, progress) {
  const node = nodeElement(nodeId);
  if (!node) return;
  node.dataset.state = nextState;
  node.querySelector(".node-state").textContent = stateLabels[nextState] || nextState;
  if (progress !== undefined) {
    node.querySelector(".node-progress i").style.width = `${progress}%`;
  }
}

function syncNodeIdentities(snapshot) {
  snapshot.nodes.forEach((item) => {
    const node = nodeElement(item.id);
    if (!node) return;
    node.querySelector(".node-key").textContent = item.short_key;
    node.querySelector(".node-key").title = item.key || "Identity pending";
    node.querySelector(".node-container").textContent = item.container;
    node.querySelector(".node-meta strong").textContent = `:${item.port}`;
    if (["idle", "preparing", "formed", "training", "complete", "failed"].includes(snapshot.status)) {
      setNode(item.id, item.state, item.progress);
    }
  });
}

function resetVisual(snapshot) {
  clearTraceVisuals();
  ui.ledgerEmpty.hidden = false;
  ui.eventCount.textContent = "0 events";
  ui.manifest.textContent = "Waiting for membership";
  ui.agreement.dataset.complete = "false";
  ui.agreementCopy.textContent = "0 / 4 identities agreed";
  ui.trainingProgress.textContent = `0 / ${snapshot.round_count || 0}`;
  ui.trainingLog.textContent = "Waiting for training.";
  snapshot.nodes.forEach((item) => setNode(item.id, "offline", 0));
  syncNodeIdentities(snapshot);
  updatePhases();
}

function clearTraceVisuals() {
  visualEpoch += 1;
  processedEvents = [];
  seenSequences = new Set();
  phaseEvidence = phaseRules.map(() => new Set());
  ui.ledger.replaceChildren();
  ui.packetLayer.replaceChildren();
  ui.routes.querySelectorAll(".active, .invitation-active").forEach((route) => {
    route.classList.remove("active", "invitation-active");
  });
}

function syncEvents(events, animate = true) {
  events
    .filter((event) => !seenSequences.has(event.sequence))
    .sort((left, right) => left.sequence - right.sequence)
    .forEach((event) => {
      seenSequences.add(event.sequence);
      processedEvents.push(event);
      applyEvent(event);
      appendLedgerEvent(event);
      if (animate) animateEnvelope(event);
    });
  updatePhases();
}

function applyEvent(event) {
  const { source, target, type } = event;
  if (type === "INVITATION") {
    setNode("node-0", "enrolling", 8);
    ["node-1", "node-2", "node-3"].forEach((nodeId) => setNode(nodeId, "invited", 5));
    return;
  }
  if (type === "JOIN_REQUEST") setNode(source, "joining", 10);
  if (type === "JOIN_ACCEPTED") setNode(target, "accepted", 15);
  if (type === "MANIFEST_SEALED") {
    setNode("node-0", "sealing", 25);
    setNode(target, "manifest", 25);
    ui.manifest.textContent = event.detail;
  }
  if (type === "TRANSFER_BEGIN") setNode(target, "syncing", 35);
  if (type === "CHUNK") setNode(target, "syncing", 75);
  if (type === "TRANSFER_COMPLETE") setNode(target, "verifying", 100);
  if (type === "READY") {
    setNode(source, "ready", 100);
    setNode("node-0", "waiting", 75);
  }
  if (type === "START") setNode(target, "starting", 100);
  if (type === "START_ACK") setNode(source, "ready", 100);
}

function updatePhases() {
  processedEvents.forEach((event) => {
    phaseRules.forEach((rule, index) => {
      if (event.type === rule.type) phaseEvidence[index].add(event[rule.key]);
    });
  });

  let completed = 0;
  let currentAssigned = false;
  ui.phaseItems.forEach((item, index) => {
    const count = Math.min(phaseEvidence[index].size, phaseRules[index].required);
    const complete = count >= phaseRules[index].required;
    item.classList.toggle("complete", complete);
    item.classList.remove("current");
    item.querySelector("b").textContent = `${count}/${phaseRules[index].required}`;
    if (complete) completed += 1;
    if (!complete && !currentAssigned && processedEvents.length) {
      item.classList.add("current");
      currentAssigned = true;
    }
  });
  ui.phaseCount.textContent = `${completed} / 6`;
}

function finalizeFormation() {
  const nodeState = backend?.status === "complete" ? "complete" : "ready";
  if (backend?.status !== "training") {
    ["node-0", "node-1", "node-2", "node-3"].forEach((nodeId) =>
      setNode(nodeId, nodeState, 100),
    );
  }
  if (backend?.manifest_hash) ui.manifest.textContent = backend.manifest_hash;
  ui.agreement.dataset.complete = "true";
  ui.agreementCopy.textContent = "4 / 4 identities agreed";
}

function appendLedgerEvent(event) {
  ui.ledgerEmpty.hidden = true;
  const item = document.createElement("li");
  item.dataset.sequence = String(event.sequence);
  item.classList.toggle("delivered", event.delivered);
  item.classList.toggle("out-of-band", event.transport === "OUT_OF_BAND");

  const time = document.createElement("time");
  time.className = "event-time";
  time.dateTime = event.timestamp;
  time.textContent = event.timestamp.slice(11, 23);

  const main = document.createElement("span");
  main.className = "event-main";
  const title = document.createElement("strong");
  title.textContent = eventLabels[event.type] || event.type;
  const route = document.createElement("span");
  route.textContent = `${shortNode(event.source)} → ${shortNode(event.target)} · ${event.transport}`;
  main.append(title, route);

  const delivery = document.createElement("span");
  delivery.className = "event-delivery";
  delivery.textContent = event.delivered ? "✓" : "·";
  delivery.title = event.delivered ? "Delivered" : "Awaiting delivery evidence";

  item.append(time, main, delivery);
  ui.ledger.prepend(item);
  while (ui.ledger.children.length > 60) ui.ledger.lastElementChild.remove();
  ui.eventCount.textContent = `${processedEvents.length} events`;
}

function syncDelivery(events) {
  events.forEach((event) => {
    const item = ui.ledger.querySelector(`[data-sequence="${event.sequence}"]`);
    if (!item) return;
    item.classList.toggle("delivered", event.delivered);
    const delivery = item.querySelector(".event-delivery");
    delivery.textContent = event.delivered ? "✓" : "·";
    delivery.title = event.delivered ? "Delivered" : "Awaiting delivery evidence";
  });
}

function shortNode(nodeId) {
  if (nodeId === "all-participants") return "N1–N3";
  return nodeId.replace("node-", "N").toUpperCase();
}

function drawRoutes() {
  if (window.innerWidth <= 860) return;
  const topologyRect = ui.topology.getBoundingClientRect();
  const nodeIds = ["node-0", "node-1", "node-2", "node-3"];
  const centers = new Map(
    nodeIds.map((nodeId) => {
      const rect = nodeElement(nodeId).getBoundingClientRect();
      return [nodeId, center(rect, topologyRect)];
    }),
  );
  ui.routes.setAttribute("viewBox", `0 0 ${topologyRect.width} ${topologyRect.height}`);
  ui.routes.replaceChildren();
  nodeIds.forEach((sourceId, sourceIndex) => {
    nodeIds.slice(sourceIndex + 1).forEach((targetId) => {
      const source = centers.get(sourceId);
      const target = centers.get(targetId);
      if (!source || !target) return;
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.id = routeId(sourceId, targetId);
      path.classList.add("route-path");
      if (Math.abs(nodeIds.indexOf(targetId) - sourceIndex) === 2) {
        path.classList.add("diagonal");
      }
      path.setAttribute("d", `M ${source.x} ${source.y} L ${target.x} ${target.y}`);
      ui.routes.append(path);
    });
  });
}

function routeId(sourceId, targetId) {
  return `route-${[sourceId, targetId].sort().join("-")}`;
}

function animateEnvelope(event) {
  const epoch = visualEpoch;
  const targets = event.target === "all-participants" ? ["node-1", "node-2", "node-3"] : [event.target];
  targets.forEach((target, index) => {
    window.setTimeout(
      () => {
        if (epoch === visualEpoch) animatePacket(event.source, target, event, epoch);
      },
      prefersReducedMotion.matches ? 0 : index * 110,
    );
  });
}

function animatePacket(sourceId, targetId, event, epoch) {
  if (epoch !== visualEpoch) return;
  const route = document.querySelector(`#${routeId(sourceId, targetId)}`);
  if (route) {
    const activeClass = event.transport === "OUT_OF_BAND" ? "invitation-active" : "active";
    route.classList.add(activeClass);
    window.setTimeout(() => {
      if (epoch === visualEpoch) route.classList.remove(activeClass);
    }, 850);
  }
  if (prefersReducedMotion.matches || window.innerWidth <= 860) return;

  const source = nodeElement(sourceId);
  const target = nodeElement(targetId);
  if (!source || !target) return;
  const topologyRect = ui.topology.getBoundingClientRect();
  const sourceRect = source.getBoundingClientRect();
  const targetRect = target.getBoundingClientRect();
  const start = center(sourceRect, topologyRect);
  const end = center(targetRect, topologyRect);
  const direction = end.y >= start.y ? 1 : -1;
  const middle = {
    x: start.x + (end.x - start.x) * 0.5,
    y: start.y + (end.y - start.y) * 0.5 + direction * 24,
  };

  const packet = document.createElement("span");
  packet.className = `packet ${packetClass(event)}`;
  packet.textContent = eventLabels[event.type] || event.type;
  ui.packetLayer.append(packet);
  const animation = packet.animate(
    [
      { opacity: 0, transform: `translate(${start.x}px, ${start.y}px) scale(0.82)` },
      { opacity: 1, offset: 0.12, transform: `translate(${start.x}px, ${start.y}px) scale(1)` },
      { opacity: 1, offset: 0.55, transform: `translate(${middle.x}px, ${middle.y}px) scale(1)` },
      { opacity: 1, offset: 0.88, transform: `translate(${end.x}px, ${end.y}px) scale(1)` },
      { opacity: 0, transform: `translate(${end.x}px, ${end.y}px) scale(0.82)` },
    ],
    { duration: 1050, easing: "cubic-bezier(0.16, 1, 0.3, 1)" },
  );
  animation.finished.then(
    () => packet.remove(),
    () => packet.remove(),
  );
}

function center(rect, parentRect) {
  return {
    x: rect.left + rect.width / 2 - parentRect.left,
    y: rect.top + rect.height / 2 - parentRect.top,
  };
}

function packetClass(event) {
  if (event.transport === "OUT_OF_BAND") return "invitation";
  if (event.type === "CHUNK") return "data";
  if (event.type === "CHUNK_ACK") return "ack";
  return "control";
}

function formatElapsed(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(2).padStart(5, "0")}`;
}

function renderRunChrome() {
  if (!backend) return;
  const roundsChanged =
    backend.status === "formed" &&
    Number(ui.roundCount.value) !== Number(backend.round_count);
  let status = backend.status;
  let detail = backend.status_detail;
  if (clientError) {
    status = "failed";
    detail = clientError;
  }
  ui.body.dataset.runStatus = status;
  ui.status.textContent = {
    idle: "Awaiting launch",
    preparing: "Preparing real AXL",
    running: "Formation live",
    formed: "Formation ready",
    training: "Training live",
    complete: "Training complete",
    failed: clientError ? "Action rejected" : "Run stopped",
  }[status];
  ui.statusDetail.textContent = detail;
  if (ui.runId) ui.runId.textContent = backend.run_id;
  if (ui.transportCopy) {
    ui.transportCopy.textContent =
      backend.deployment === "docker"
        ? "4 Docker node containers · zero in-memory hops"
        : "4 local bridges · zero in-memory hops";
  }
  if (ui.elapsed) ui.elapsed.textContent = formatElapsed(backend.elapsed_seconds);
  const formationBusy =
    backend.status === "preparing" ||
    backend.status === "running" ||
    backend.status === "training" ||
    requestInFlight !== null ||
    pendingRequest !== null;
  ui.start.disabled =
    formationBusy || backend.can_begin_formation === false;
  ui.startLabel.textContent =
    backend.status === "formed"
      ? roundsChanged
        ? "Apply rounds"
        : "Re-form group"
      : backend.status === "complete"
        ? "Run again"
        : "Begin formation";
  ui.startBusyLabel.textContent =
    backend.status === "formed" || backend.status === "training"
      ? "Group formed"
      : backend.status === "complete"
        ? "Training complete"
        : "Formation running";
  ui.startTraining.disabled =
    backend.deployment !== "docker" ||
    backend.status !== "formed" ||
    requestInFlight !== null ||
    pendingRequest !== null ||
    backend.can_start_training === false ||
    roundsChanged;
  ui.startTraining.textContent =
    backend.status === "training"
      ? "Training running"
      : backend.status === "complete"
        ? "Training complete"
        : "Start training";
  ui.roundCount.disabled = ["preparing", "running", "training"].includes(
    backend.status,
  );
}

function renderTraining(snapshot) {
  const training = snapshot.training || {};
  const completed = Number(training.completed_rounds || 0);
  const total = Number(training.round_count || snapshot.round_count || 0);
  ui.trainingProgress.textContent = `${completed} / ${total}`;
  const logs = Array.isArray(training.logs) ? training.logs : [];
  ui.trainingLog.textContent = logs.length
    ? logs.map((log) => `${log.timestamp.slice(11, 23)} ${log.node}  ${log.message}`).join("\n")
    : "Waiting for training.";
  ui.trainingLog.scrollTop = ui.trainingLog.scrollHeight;
}

function showError(snapshot) {
  if (!snapshot.error || shownErrorGeneration === snapshot.generation) return;
  shownErrorGeneration = snapshot.generation;
  ui.errorMessage.textContent = snapshot.error;
  ui.errorDialog.showModal();
}

async function poll() {
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (!response.ok) throw new Error(`Dashboard state returned ${response.status}`);
    const snapshot = await response.json();
    backend = snapshot;
    if (generation !== snapshot.generation) {
      clientError = null;
      pendingRequest = null;
      generation = snapshot.generation;
      ui.roundCount.value = String(snapshot.round_count);
      resetVisual(snapshot);
      syncEvents(snapshot.events, false);
    }
    if (
      pendingRequest === "training" &&
      ["training", "complete", "failed"].includes(snapshot.status)
    ) {
      pendingRequest = null;
    }
    syncNodeIdentities(snapshot);
    syncEvents(snapshot.events);
    syncDelivery(snapshot.events);
    if (["formed", "training", "complete"].includes(snapshot.status)) {
      finalizeFormation();
    }
    if (snapshot.status === "failed") showError(snapshot);
    if (snapshot.status === "failed") clientError = null;
    renderRunChrome();
    renderTraining(snapshot);
  } catch (error) {
    ui.body.dataset.runStatus = "failed";
    ui.status.textContent = "Dashboard disconnected";
    ui.statusDetail.textContent = error.message;
  } finally {
    window.setTimeout(poll, 250);
  }
}

async function startRun() {
  if (requestInFlight !== null) return;
  clientError = null;
  ui.start.disabled = true;
  requestInFlight = "formation";
  try {
    const roundCount = Number(ui.roundCount.value);
    if (!Number.isInteger(roundCount) || roundCount < 1 || roundCount > 100) {
      throw new Error("Training rounds must be an integer from 1 to 100");
    }
    const response = await fetch("/api/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ round_count: roundCount }),
    });
    const result = await response.json().catch(() => ({}));
    if (!response.ok || result.started !== true) {
      throw new Error(
        result.error || `Start request returned ${response.status}`,
      );
    }
    clearTraceVisuals();
    ui.ledgerEmpty.hidden = false;
    ui.eventCount.textContent = "0 events";
    ui.manifest.textContent = "Waiting for membership";
    ui.agreement.dataset.complete = "false";
    ui.agreementCopy.textContent = "0 / 4 identities agreed";
    ui.trainingProgress.textContent = `0 / ${roundCount}`;
    ui.trainingLog.textContent = "Waiting for training.";
    if (backend) {
      backend = {
        ...backend,
        status: "preparing",
        status_detail: "Formation request accepted",
        round_count: roundCount,
      };
    }
    ui.body.dataset.runStatus = "preparing";
    ui.status.textContent = "Preparing real AXL";
    ui.statusDetail.textContent = "Formation request accepted";
    pendingRequest = "formation";
  } catch (error) {
    pendingRequest = null;
    clientError = `Could not start formation: ${error.message}`;
  } finally {
    requestInFlight = null;
    if (backend) renderRunChrome();
  }
}

async function startTraining() {
  if (requestInFlight !== null) return;
  clientError = null;
  ui.startTraining.disabled = true;
  requestInFlight = "training";
  try {
    const response = await fetch("/api/train", { method: "POST" });
    const result = await response.json().catch(() => ({}));
    if (!response.ok || result.started !== true) {
      throw new Error(
        result.error || `Training request returned ${response.status}`,
      );
    }
    pendingRequest = "training";
    if (backend) {
      backend = {
        ...backend,
        status: "training",
        status_detail: "Training request accepted",
      };
    }
  } catch (error) {
    pendingRequest = null;
    clientError = `Could not start training: ${error.message}`;
  } finally {
    requestInFlight = null;
    if (backend) renderRunChrome();
  }
}

ui.start.addEventListener("click", startRun);
ui.startTraining.addEventListener("click", startTraining);
ui.roundCount.addEventListener("input", renderRunChrome);
window.addEventListener("resize", drawRoutes);
window.addEventListener("load", drawRoutes);
drawRoutes();
poll();
