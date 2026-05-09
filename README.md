# telemt fork for mtconf

This repository is a trimmed production fork used by `mtconf`.

It keeps the Rust source, tests, lockfile, and license needed to build the
`mtproxy` service from local sources. Standalone deployment docs, release
automation, example compose files, helper scripts, and public-project metadata
were intentionally removed to avoid confusion: deployment, configuration,
dashboards, and operations live in the parent `mtconf` repository.

## Build

```bash
cargo build --release
```

The production image is built by the parent repository Dockerfile.
