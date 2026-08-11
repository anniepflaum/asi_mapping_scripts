# ASI mapping data manifests

These catalogs were extracted from the checksum-complete lab317 inventory
created on 2026-08-07. They preserve the pre-migration relative path, size,
modification timestamp, and SHA-256 for each external data area.

The image archive and mapped products were moved on the same filesystem and
then re-verified against every manifest entry. No compatibility symlinks were
created; `core/paths.py` now defaults to the shared data and output roots.
