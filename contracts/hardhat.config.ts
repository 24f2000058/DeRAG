import { HardhatUserConfig } from "hardhat/config";
import "@nomicfoundation/hardhat-toolbox";

// Hardhat's built-in "hardhat" network (and its "localhost" mirror, once
// `npx hardhat node` is running) auto-generates 20 deterministic funded
// accounts by default. Per the DeRAG spec: Account #0 is reserved as
// Admin/Oracle, Accounts #1-#5 as publishers. No code assigns these roles
// yet — that lands in Milestone 3 alongside DeRAG_Registry.sol.
const config: HardhatUserConfig = {
  solidity: "0.8.20",
  networks: {
    localhost: {
      url: "http://127.0.0.1:8545",
      chainId: 31337,
    },
  },
};

export default config;
