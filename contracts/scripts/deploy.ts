import { ethers, artifacts } from "hardhat";
import * as fs from "fs";
import * as path from "path";

async function main() {
  const Registry = await ethers.getContractFactory("DeRAG_Registry");
  const registry = await Registry.deploy();
  await registry.waitForDeployment();
  const address = await registry.getAddress();

  const artifact = await artifacts.readArtifact("DeRAG_Registry");

  const deploymentsDir = path.join(__dirname, "..", "deployments");
  fs.mkdirSync(deploymentsDir, { recursive: true });
  fs.writeFileSync(
    path.join(deploymentsDir, "localhost.json"),
    JSON.stringify({ address, abi: artifact.abi }, null, 2)
  );

  console.log(`DeRAG_Registry deployed to ${address}`);
  console.log("Deployment artifact written to contracts/deployments/localhost.json");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
