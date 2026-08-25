import json
import os
from pathlib import Path

import requests
from eth_account import Account
from web3 import Web3
from web3.contract import Contract
from web3.exceptions import ContractLogicError

DEPLOYMENT_PATH = (
    Path(__file__).resolve().parent.parent / "contracts" / "deployments" / "localhost.json"
)


class RegistryError(Exception):
    pass


def _doc_hash_to_bytes32(doc_hash: str) -> bytes:
    hexstr = doc_hash if doc_hash.startswith("0x") else "0x" + doc_hash
    return Web3.to_bytes(hexstr=hexstr)


def connect() -> tuple[Web3, Contract]:
    provider_uri = os.environ["WEB3_PROVIDER_URI"]
    try:
        w3 = Web3(Web3.HTTPProvider(provider_uri))
        with open(DEPLOYMENT_PATH) as f:
            deployment = json.load(f)
        contract = w3.eth.contract(address=deployment["address"], abi=deployment["abi"])
    except (requests.exceptions.RequestException, FileNotFoundError, ValueError) as exc:
        raise RegistryError(f"could not connect to blockchain node or read deployment: {exc}") from exc
    return w3, contract


def _send_transaction(w3: Web3, function_call, signing_key: str) -> dict:
    try:
        account = Account.from_key(signing_key)
        tx = function_call.build_transaction(
            {
                "from": account.address,
                "nonce": w3.eth.get_transaction_count(account.address),
                "gasPrice": w3.eth.gas_price,
                "chainId": w3.eth.chain_id,
            }
        )
        signed = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    except (requests.exceptions.RequestException, ContractLogicError, ValueError) as exc:
        raise RegistryError(f"blockchain transaction failed: {exc}") from exc
    return dict(receipt)


def register_document(doc_hash: str, ipfs_cid: str, signing_key: str | None = None) -> dict:
    w3, contract = connect()
    key = signing_key or os.environ["PRIVATE_KEY"]
    hash_bytes = _doc_hash_to_bytes32(doc_hash)
    function_call = contract.functions.registerDocument(hash_bytes, ipfs_cid)
    return _send_transaction(w3, function_call, key)


def update_reputation(doc_hash: str, delta: int, signing_key: str | None = None) -> dict:
    w3, contract = connect()
    key = signing_key or os.environ["PRIVATE_KEY"]
    hash_bytes = _doc_hash_to_bytes32(doc_hash)
    function_call = contract.functions.updateReputation(hash_bytes, delta)
    return _send_transaction(w3, function_call, key)


def verify_integrity(doc_hash: str) -> dict:
    w3, contract = connect()
    hash_bytes = _doc_hash_to_bytes32(doc_hash)
    try:
        exists, reputation, publisher = contract.functions.verifyIntegrity(hash_bytes).call()
    except (requests.exceptions.RequestException, ContractLogicError, ValueError) as exc:
        raise RegistryError(f"could not read document registry: {exc}") from exc
    return {"exists": exists, "reputation": reputation, "publisher": publisher}


def get_document(doc_hash: str) -> dict:
    w3, contract = connect()
    hash_bytes = _doc_hash_to_bytes32(doc_hash)
    try:
        publisher, content_hash, ipfs_cid, reputation_score, timestamp, is_registered = (
            contract.functions.documentLedger(hash_bytes).call()
        )
    except (requests.exceptions.RequestException, ContractLogicError, ValueError) as exc:
        raise RegistryError(f"could not read document registry: {exc}") from exc
    return {
        "publisher": publisher,
        "content_hash": "0x" + content_hash.hex(),
        "ipfs_cid": ipfs_cid,
        "reputation_score": reputation_score,
        "timestamp": timestamp,
        "is_registered": is_registered,
    }
