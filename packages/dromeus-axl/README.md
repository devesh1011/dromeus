# dromeus-axl

A prebuilt [AXL](https://github.com/gensyn-ai/axl) network node for
[Dromeus](https://github.com/devesh1011/dromeus). `pip install dromeus` installs
it, so you don't need Go or an AXL source checkout.

AXL is Gensyn's software, redistributed here with Gensyn's permission. Each
release pins one AXL commit, built with `CGO_ENABLED=0 go build -trimpath`.

| Platform | Wheel |
| --- | --- |
| Linux x86_64 | `manylinux_2_17_x86_64` |
| Linux aarch64 | `manylinux_2_17_aarch64` |
| macOS 12+ on Apple silicon | `macosx_12_0_arm64` |

## Use

```bash
dromeus-axl -config axl.json      # runs the bundled AXL node
```

```python
import dromeus_axl

dromeus_axl.binary()       # path of the executable
dromeus_axl.build_info()   # commit, Go version, target, sha256
```

## Build

From the Dromeus repository, with Go and uv installed:

```bash
python scripts/build_axl_wheels.py
```

Wheels land in `dist/axl/`. There is no source distribution.
