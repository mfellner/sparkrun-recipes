#!/usr/bin/env python3
"""Fail-closed RoCEv2 GID/netdev/IPv4 validation for selected HCAs."""
from __future__ import annotations

import argparse
import fcntl
import ipaddress
import json
import re
import socket
import struct
from pathlib import Path
from typing import Callable

NAME = re.compile(r"^[A-Za-z0-9_.:-]+$")


def assigned_global_ipv4(netdev: str) -> list[str]:
    """Read the exact interface's primary IPv4 without external image tools."""
    encoded = netdev.encode("ascii")
    if not encoded or len(encoded) >= 16:
        raise RuntimeError(f"invalid Linux interface name length: {netdev!r}")
    request = struct.pack("256s", encoded)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            response = fcntl.ioctl(sock.fileno(), 0x8915, request)  # SIOCGIFADDR
    except OSError as exc:
        raise RuntimeError(f"SIOCGIFADDR failed for {netdev}: {exc}") from exc
    address = ipaddress.IPv4Address(socket.inet_ntoa(response[20:24]))
    if address.is_unspecified or address.is_loopback or address.is_link_local or address.is_multicast:
        raise RuntimeError(f"{netdev} has non-routable IPv4 {address}")
    return [str(address)]


def verify_hcas(
    sys_root: Path,
    gid_index: int,
    hcas: list[str],
    ipv4_lookup: Callable[[str], list[str]] = assigned_global_ipv4,
) -> list[str]:
    """Return all per-HCA validation failures; an empty list is success."""
    failures: list[str] = []
    if not 0 <= gid_index <= 255:
        return [f"GID index {gid_index} is outside supported range 0..255"]
    if not hcas:
        return ["NCCL_IB_HCA is empty"]
    if len(set(hcas)) != len(hcas):
        failures.append("NCCL_IB_HCA contains duplicate HCAs")
    for hca in hcas:
        if not hca or not NAME.fullmatch(hca):
            failures.append(f"invalid HCA name: {hca!r}")
            continue
        port = sys_root / "class/infiniband" / hca / "ports/1"
        paths = {
            "gid": port / "gids" / str(gid_index),
            "type": port / "gid_attrs/types" / str(gid_index),
            "ndev": port / "gid_attrs/ndevs" / str(gid_index),
        }
        values: dict[str, str] = {}
        missing = False
        for key, path in paths.items():
            try:
                values[key] = path.read_text().strip()
            except OSError as exc:
                failures.append(f"{hca}: missing/unreadable {key} path {path}: {exc}")
                missing = True
        if missing:
            continue
        if values["type"] != "RoCE v2":
            failures.append(
                f"{hca}: GID index {gid_index} type is {values['type']!r}, expected 'RoCE v2'"
            )
        netdev = values["ndev"]
        if not netdev or not NAME.fullmatch(netdev):
            failures.append(f"{hca}: invalid ndev for GID index {gid_index}: {netdev!r}")
            continue
        try:
            gid = ipaddress.IPv6Address(values["gid"])
        except ValueError:
            failures.append(
                f"{hca}: invalid IPv6 GID at index {gid_index}: {values['gid']!r}"
            )
            continue
        if int(gid) == 0:
            failures.append(f"{hca}: GID index {gid_index} is zero")
            continue
        if gid.ipv4_mapped is None:
            failures.append(
                f"{hca}: GID index {gid_index} is not an IPv4-mapped IPv6 address"
            )
            continue
        mapped = str(gid.ipv4_mapped)
        try:
            addresses = ipv4_lookup(netdev)
        except Exception as exc:  # fail closed with the query reason
            failures.append(f"{hca}: cannot read global IPv4 for {netdev}: {exc}")
            continue
        if len(addresses) != 1:
            failures.append(
                f"{hca}: ndev {netdev} must have exactly one assigned global IPv4, got {addresses!r}"
            )
            continue
        try:
            assigned = str(ipaddress.IPv4Address(addresses[0]))
        except ValueError:
            failures.append(f"{hca}: ndev {netdev} has invalid IPv4 {addresses[0]!r}")
            continue
        if mapped != assigned:
            failures.append(
                f"{hca}: GID index {gid_index} maps {mapped}, does not map to {netdev} IPv4 {assigned}"
            )
    return failures


def capture_hcas(
    sys_root: Path,
    gid_index: int,
    hcas: list[str],
    ipv4_lookup: Callable[[str], list[str]] = assigned_global_ipv4,
) -> list[dict[str, object]]:
    """Return canonical records only after the complete fail-closed gate passes."""
    failures = verify_hcas(sys_root, gid_index, hcas, ipv4_lookup)
    if failures:
        raise RuntimeError("; ".join(failures))
    records: list[dict[str, object]] = []
    for hca in hcas:
        port = sys_root / "class/infiniband" / hca / "ports/1"
        gid = ipaddress.IPv6Address((port / "gids" / str(gid_index)).read_text().strip())
        gid_type = (port / "gid_attrs/types" / str(gid_index)).read_text().strip()
        netdev = (port / "gid_attrs/ndevs" / str(gid_index)).read_text().strip()
        ipv4 = str(ipaddress.IPv4Address(ipv4_lookup(netdev)[0]))
        records.append({
            "hca": hca,
            "gid_index": gid_index,
            "gid": str(gid),
            "gid_type": gid_type,
            "netdev": netdev,
            "ipv4": ipv4,
        })
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sys-root", type=Path, default=Path("/sys"))
    parser.add_argument("--gid-index", type=int, required=True)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("hcas", nargs="+")
    args = parser.parse_args()
    failures = verify_hcas(args.sys_root, args.gid_index, args.hcas)
    if failures:
        for failure in failures:
            print(f"FATAL: {failure}", file=__import__("sys").stderr)
        return 1
    if args.json:
        print(json.dumps(capture_hcas(args.sys_root, args.gid_index, args.hcas), sort_keys=True))
    else:
        for hca in args.hcas:
            print(f"[OK] {hca} GID {args.gid_index} is RoCE v2 and maps its netdev IPv4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
