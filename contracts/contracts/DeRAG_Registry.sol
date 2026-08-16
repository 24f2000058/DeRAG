// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract DeRAG_Registry {
    struct Document {
        address publisher;
        bytes32 contentHash;
        string ipfsCID;
        int256 reputationScore;
        uint256 timestamp;
        bool isRegistered;
    }

    address public owner;
    mapping(address => bool) public authorizedOracles;
    mapping(bytes32 => Document) public documentLedger;

    modifier onlyOwner() {
        require(msg.sender == owner, "DeRAG_Registry: caller is not the owner");
        _;
    }

    modifier onlyAuthorizedOracle() {
        require(authorizedOracles[msg.sender], "DeRAG_Registry: caller is not an authorized oracle");
        _;
    }

    constructor() {
        owner = msg.sender;
        authorizedOracles[msg.sender] = true;
    }

    function authorizeOracle(address _account, bool _authorized) external onlyOwner {
        authorizedOracles[_account] = _authorized;
    }

    function registerDocument(bytes32 _hash, string memory _cid) external {
        require(!documentLedger[_hash].isRegistered, "DeRAG_Registry: document already registered");
        documentLedger[_hash] = Document({
            publisher: msg.sender,
            contentHash: _hash,
            ipfsCID: _cid,
            reputationScore: 0,
            timestamp: block.timestamp,
            isRegistered: true
        });
    }

    function updateReputation(bytes32 _hash, int256 _delta) external onlyAuthorizedOracle {
        require(documentLedger[_hash].isRegistered, "DeRAG_Registry: document not registered");
        documentLedger[_hash].reputationScore += _delta;
    }

    function verifyIntegrity(bytes32 _hash)
        external
        view
        returns (bool exists, int256 reputation, address publisher)
    {
        Document storage doc = documentLedger[_hash];
        return (doc.isRegistered, doc.reputationScore, doc.publisher);
    }
}
