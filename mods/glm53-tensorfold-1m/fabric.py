#!/usr/bin/env python3
"""Fail closed unless both selected TensorFold RoCE rails reach the pinned peer."""
import fcntl
import ipaddress
import os
from pathlib import Path
import socket
import struct
import sys
import time


def ipv4_ioctl(dev: str, request: int) -> ipaddress.IPv4Address:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        result = fcntl.ioctl(sock.fileno(), request, struct.pack("256s", dev.encode()[:15]))
    return ipaddress.IPv4Address(result[20:24])


def checksum(payload: bytes) -> int:
    if len(payload) % 2:
        payload += b"\0"
    words = struct.unpack(f"!{len(payload) // 2}H", payload)
    total = sum(words)
    while total >> 16:
        total = (total & 0xffff) + (total >> 16)
    return (~total) & 0xffff


def peer_probe(dev: str, local: ipaddress.IPv4Address, peer: ipaddress.IPv4Address) -> None:
    """ICMP echo bound to BOTH source IP and netdev, without image 'ping'."""
    nonce = os.urandom(8)
    ident = os.getpid() & 0xffff
    for sequence in (1, 2):
        payload = struct.pack("!BBHHH", 8, 0, 0, ident, sequence) + nonce
        payload = struct.pack("!BBHHH", 8, 0, checksum(payload), ident, sequence) + nonce
        with socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, dev.encode() + b"\0")
            sock.bind((str(local), 0))
            sock.settimeout(1.5)
            sock.sendto(payload, (str(peer), 0))
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                try:
                    packet, address = sock.recvfrom(4096)
                except socket.timeout:
                    break
                ihl = (packet[0] & 0x0f) * 4
                if (address[0] == str(peer) and len(packet) >= ihl + 16 and
                        packet[ihl] == 0 and packet[ihl + 1] == 0 and
                        packet[ihl + 4:ihl + 8] == struct.pack("!HH", ident, sequence) and
                        packet[ihl + 8:ihl + 16] == nonce):
                    return
    raise ValueError(f"no ICMP response from {peer} via {local}/{dev}")


def check_rail(hca: str, gid_text: str, local: ipaddress.IPv4Address,
               peer: ipaddress.IPv4Address) -> str:
    hca_dir = Path("/sys/class/infiniband") / hca
    port = hca_dir / "ports/1"
    if not (port / "state").read_text().strip().endswith("ACTIVE"):
        raise ValueError(f"{hca} is not ACTIVE")
    if not (port / "phys_state").read_text().strip().endswith("LinkUp"):
        raise ValueError(f"{hca} physical link is not up")
    gid = ipaddress.IPv6Address((port / "gids" / gid_text).read_text().strip())
    kind = (port / "gid_attrs/types" / gid_text).read_text().strip()
    if gid.ipv4_mapped != local or "v2" not in kind:
        raise ValueError(f"{hca} GID {gid_text} does not map to {local} with RoCEv2")
    matches = []
    for netdev in os.listdir("/sys/class/net"):
        infiniband = Path("/sys/class/net") / netdev / "device/infiniband" / hca
        if not infiniband.exists():
            continue
        try:
            address = ipv4_ioctl(netdev, 0x8915)
            mask = ipv4_ioctl(netdev, 0x891B)
        except OSError:
            continue
        if address == local:
            network = ipaddress.IPv4Network(f"{address}/{mask}", strict=False)
            if peer in network and (port / "gid_attrs/ndevs" / gid_text).read_text().strip() == netdev:
                if (Path("/sys/class/net") / netdev / "operstate").read_text().strip() == "up":
                    matches.append(netdev)
    if len(matches) != 1:
        raise ValueError(f"{hca} does not uniquely map {local} to peer {peer} on an up RoCE netdev")
    peer_probe(matches[0], local, peer)
    return matches[0]


def check(rank: str, master_text: str, node_text: str, hcas_text: str, gid_text: str) -> str:
    if rank not in {"0", "1"} or not gid_text.isdigit():
        raise ValueError("invalid rank or GID index")
    master = ipaddress.IPv4Address(master_text)
    worker = ipaddress.IPv4Address(os.environ["TF_FABRIC_WORKER"])
    secondary_master = ipaddress.IPv4Address(os.environ["TF_FABRIC_MASTER_SECONDARY"])
    secondary_worker = ipaddress.IPv4Address(os.environ["TF_FABRIC_WORKER_SECONDARY"])
    local = ipaddress.IPv4Address(node_text)
    if master == worker or secondary_master == secondary_worker:
        raise ValueError("fabric peer address must differ")
    expected_local = master if rank == "0" else worker
    if local != expected_local:
        raise ValueError("rank/primary fabric address mismatch")
    selected = [hca.strip() for hca in hcas_text.split(",")]
    if len(selected) != 2 or len(set(selected)) != 2 or any(not hca.isidentifier() for hca in selected):
        raise ValueError("NCCL_IB_HCA must list exactly two distinct local RoCE HCAs")
    primary_peer = worker if rank == "0" else master
    secondary_local = secondary_master if rank == "0" else secondary_worker
    secondary_peer = secondary_worker if rank == "0" else secondary_master
    first = check_rail(selected[0], gid_text, local, primary_peer)
    check_rail(selected[1], gid_text, secondary_local, secondary_peer)
    return first


if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("usage: fabric.py rank master node hcas gid_index")
    try:
        print(check(*sys.argv[1:]))
    except (ValueError, OSError, KeyError) as error:
        sys.exit(f"invalid SparkRun fabric: {error}")
