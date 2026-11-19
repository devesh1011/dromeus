# AXL setup

Every Dromeus node talks to the network through [AXL](https://github.com/gensyn-ai/axl),
Gensyn's peer-to-peer network node. AXL runs as a separate process on each
machine, encrypts and routes traffic between machines, and gives Dromeus a local
HTTP bridge. It runs in user space: no TUN interface, no root. This guide sets up
AXL for a group whose machines are on different networks; to try Dromeus on one
machine first, follow the [quickstart](quickstart.md).

## How the pieces connect

- Each machine runs one AXL node and one Dromeus node.
- Dromeus reaches its AXL node over HTTP on loopback, at the `axl_bridge_url` in
  its [node configuration](configuration.md#node-configuration).
- AXL nodes link to each other over TLS. A node needs a link to only one other
  node in the group; AXL routes traffic onward. In the quickstart, three nodes
  link only to the fourth and still train together.
- At least one node must accept incoming links: the group's **bootstrap peer**.
  Any participant with an open port can be one. Dromeus hosts nothing.

## 1. Get AXL

On Linux (x86_64, aarch64) and macOS on Apple silicon, installing Dromeus
installs a pinned AXL build as the `dromeus-axl` command. On other platforms,
[build it from source](#other-platforms).

## 2. Create a persistent identity

An AXL node's identity is an ed25519 key. Without a key file, AXL makes a new
identity in memory every time it starts. That's fine for a one-machine trial, but
in a real group each node's public key is sealed into the run's manifest, so a
node must keep the same key for the whole run.

On Linux:

```bash
openssl genpkey -algorithm ed25519 -out private.pem
chmod 600 private.pem
```

On macOS, the system `openssl` (LibreSSL) can't generate ed25519 keys; use
OpenSSL from Homebrew:

```bash
brew install openssl
"$(brew --prefix openssl)/bin/openssl" genpkey -algorithm ed25519 -out private.pem
chmod 600 private.pem
```

Keep `private.pem` private: anyone with it can act as your node.

## 3. Write the AXL configuration

On the **bootstrap peer**, the machine that accepts incoming links. It needs a
public address, and its firewall must allow inbound TCP on the listen port:

```json
{
  "PrivateKeyPath": "private.pem",
  "Peers": [],
  "Listen": ["tls://0.0.0.0:9001"],
  "api_port": 9002,
  "bridge_addr": "127.0.0.1"
}
```

On an **outbound-only** node, such as a laptop or a machine behind a home router.
It connects out to the bootstrap peer and receives data over the same connection,
so it needs no port forwarding:

```json
{
  "PrivateKeyPath": "private.pem",
  "Peers": ["tls://BOOTSTRAP_PEER_ADDRESS:9001"],
  "Listen": [],
  "api_port": 9002,
  "bridge_addr": "127.0.0.1"
}
```

Replace `BOOTSTRAP_PEER_ADDRESS` with the bootstrap peer's public IP address or
host name. To survive one bootstrap peer going down, run more than one and list
each in `Peers`.

| Field | Meaning | Default |
| --- | --- | --- |
| `PrivateKeyPath` | The node's ed25519 key | A new key in memory at every start |
| `Peers` | Nodes to connect out to, as `tls://host:port` | None |
| `Listen` | Addresses to accept incoming links on | None |
| `api_port` | Port of the HTTP bridge Dromeus uses | `9002` |
| `bridge_addr` | Address the HTTP bridge binds to | `127.0.0.1` |
| `max_message_size` | Largest message, in bytes | `16777216` (16 MiB) |

AXL has more settings; see its
[configuration reference](https://github.com/gensyn-ai/axl/blob/main/docs/configuration.md).

Keep `bridge_addr` on `127.0.0.1`: anything that can reach the bridge can send and
receive as your node. Dromeus only accepts an `axl_bridge_url` on loopback.

## 4. Start AXL and check it

```bash
dromeus-axl -config axl.json
```

Keep it running for the whole run. Point the node configuration's
`axl_bridge_url` at the bridge (`http://127.0.0.1:9002` here), then:

```bash
dromeus doctor --config node.yaml
```

`doctor` checks that the bridge answers, that it reports this node's public key,
and how many peer links are up. An outbound-only node reports one link, to its
bootstrap peer. See the [command-line guide](command-line.md) for its exit codes.

## Pinned AXL build

A group runs one AXL build. The draft records it in `environment.axl_version`,
and `make_draft` fills it in from the bundled build, currently AXL commit
`628e28ace077f26dfe8d0259009b357216a9d8d4`. `dromeus doctor` warns when the draft
names a different build from the one installed. It can't see which build the
running AXL process is, so make sure every machine runs the build the draft names.

## Other platforms

Where no AXL build is bundled, build the pinned commit with Go 1.25.5 or later:

```bash
git clone https://github.com/gensyn-ai/axl.git
cd axl
git checkout 628e28ace077f26dfe8d0259009b357216a9d8d4
go build -o node ./cmd/node
./node -config axl.json
```

Whoever writes the draft on such a machine must name the build, because there is
no bundled one to fill it in: `dromeus draft ... --axl-version 628e28ace077f26dfe8d0259009b357216a9d8d4`.
