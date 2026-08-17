import os


def _api_base_url() -> str:
    multiaddr = os.environ["IPFS_API_URL"]
    _, protocol, host, transport, port = multiaddr.split("/")
    assert protocol == "ip4" and transport == "tcp"
    return f"http://{host}:{port}"
