// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// Trivial stub so `npx hardhat compile` / `npx hardhat test` succeed in
// Milestone 0. Replaced by DeRAG_Registry.sol in Milestone 3.
contract Placeholder {
    uint256 private value;

    function setValue(uint256 _value) external {
        value = _value;
    }

    function getValue() external view returns (uint256) {
        return value;
    }
}
