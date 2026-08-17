import os

import requests


class IPFSError(Exception):
    pass


def _api_base_url() -> str:
    multiaddr = os.environ["IPFS_API_URL"]
    _, protocol, host, transport, port = multiaddr.split("/")
    assert protocol == "ip4" and transport == "tcp"
    return f"http://{host}:{port}"


def upload_text(text: str) -> str:
    url = f"{_api_base_url()}/api/v0/add"
    try:
        response = requests.post(
            url,
            files={"file": text.encode("utf-8")},
            timeout=10,
        )
    except requests.exceptions.RequestException as exc:
        raise IPFSError(f"could not reach IPFS daemon at {url}: {exc}") from exc

    if response.status_code != 200:
        raise IPFSError(
            f"IPFS add failed with status {response.status_code}: {response.text}"
        )

    return response.json()["Hash"]


def _gateway_base_url() -> str:
    host = _api_base_url().split("://")[1].split(":")[0]
    return f"http://{host}:8080"


def fetch_text(cid: str) -> str:
    url = f"{_gateway_base_url()}/ipfs/{cid}"
    try:
        response = requests.get(url, timeout=10)
    except requests.exceptions.RequestException as exc:
        raise IPFSError(f"could not reach IPFS gateway at {url}: {exc}") from exc

    if response.status_code != 200:
        raise IPFSError(
            f"IPFS fetch failed with status {response.status_code} for CID {cid}: {response.text}"
        )

    return response.content.decode("utf-8")
