# CIFAR-10 benchmark report: PASS

- run: `dpsgd-resnet32-400-20260730-seed17`
- manifest hash: `c8f5f7592b4d78fdeda8507f997386b0d373f6d0e47e21c433c56316a656466d`
- nodes: 4
- D-PSGD final accuracy mean: 0.916900
- FedAvg final accuracy: 0.917100
- absolute accuracy gate (all nodes >= 90%): PASS
- publication ready: yes

[Accuracy and loss curves](metrics.png)

[Approximate consensus distance](consensus.png)

[AXL latency and round timing](timing.png)

[AXL payload goodput](goodput.png)

Final approximate consensus distance: 0.0

[Raw-data provenance](provenance.json)
