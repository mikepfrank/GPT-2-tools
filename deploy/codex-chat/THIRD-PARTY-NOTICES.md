# Seccomp profile attribution

`seccomp.json` is derived from Moby's default seccomp profile at the pinned
`v25.0.13` release:

- [Original profile](https://github.com/moby/moby/blob/v25.0.13/profiles/seccomp/default.json)
- [Upstream license](https://github.com/moby/moby/blob/v25.0.13/LICENSE)
- [Upstream notice](https://github.com/moby/moby/blob/v25.0.13/NOTICE)

Moby/Docker distributes this source under the Apache License, Version 2.0.
The complete pinned upstream license and notice are retained as
[`LICENSE.moby`](LICENSE.moby) and [`NOTICE.moby`](NOTICE.moby), including their
Docker copyright notices. Keep these files with redistributed profile copies.

This application's modification adds one unconditional allow rule for eight
syscalls: `clone`, `unshare`, `mount`, `umount`, `umount2`, `pivot_root`, `setns`,
and `sethostname`. The rule enables unprivileged nested user/mount namespaces
for Codex's native Bubblewrap sandbox inside an already isolated Docker
container. The profile also adds explanatory comments. The container retains
`--cap-drop ALL` and `no-new-privileges`; the modification grants no host
capability and does not use an unconfined seccomp profile.

Pinned copies fetched October 5, 2026 CDT / October 6 UTC:

| File | SHA-256 |
| --- | --- |
| `LICENSE.moby` | `7c87873291f289713ac5df48b1f2010eb6963752bbd6b530416ab99fc37914a8` |
| `NOTICE.moby` | `b40ec5b16182103ef1cd69d42a21c98369e35bd483b3973902da4807c0755446` |
