# CIFAR-10 benchmark report: PASS

- run: `dpsgd-resnet32-400-20260730-seed41`
- manifest hash: `f0de1016a77380493bedb2844fd53483b6274b0b2dc393bd4e409deeb37ec19b`
- nodes: 4
- D-PSGD final accuracy mean: 0.916000
- FedAvg final accuracy: 0.912300
- absolute accuracy gate (all nodes >= 90%): PASS
- publication ready: yes

[Accuracy and loss curves](metrics.png)

[Approximate consensus distance](consensus.png)

[AXL latency and round timing](timing.png)

[AXL payload goodput](goodput.png)

Final approximate consensus distance: 0.0

[Raw-data provenance](provenance.json)
