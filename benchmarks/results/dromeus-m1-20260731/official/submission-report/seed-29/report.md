# CIFAR-10 benchmark report: PASS

- run: `dpsgd-resnet32-400-20260730-seed29`
- manifest hash: `ec9b27e700242c8af87125ca792321ed47a0095efd8e09deb531d54e06f3f70d`
- nodes: 4
- D-PSGD final accuracy mean: 0.914400
- FedAvg final accuracy: 0.915500
- absolute accuracy gate (all nodes >= 90%): PASS
- publication ready: yes

[Accuracy and loss curves](metrics.png)

[Approximate consensus distance](consensus.png)

[AXL latency and round timing](timing.png)

[AXL payload goodput](goodput.png)

Final approximate consensus distance: 0.002430927079518334

[Raw-data provenance](provenance.json)
