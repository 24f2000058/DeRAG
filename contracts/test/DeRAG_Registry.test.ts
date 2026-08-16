import { expect } from "chai";
import { ethers } from "hardhat";

describe("DeRAG_Registry", function () {
  async function deployFixture() {
    const [owner, publisher1, publisher2, oracle2] = await ethers.getSigners();
    const Registry = await ethers.getContractFactory("DeRAG_Registry");
    const registry = await Registry.deploy();
    return { registry, owner, publisher1, publisher2, oracle2 };
  }

  const HASH_1 = ethers.keccak256(ethers.toUtf8Bytes("doc-1"));
  const HASH_UNREGISTERED = ethers.keccak256(ethers.toUtf8Bytes("never-registered"));

  it("sets publisher to msg.sender and reputation to 0 on successful registration", async function () {
    const { registry, publisher1 } = await deployFixture();
    await registry.connect(publisher1).registerDocument(HASH_1, "cid-1");
    const doc = await registry.documentLedger(HASH_1);
    expect(doc.publisher).to.equal(publisher1.address);
    expect(doc.reputationScore).to.equal(0);
    expect(doc.isRegistered).to.equal(true);
  });

  it("reverts when registering an already-registered hash", async function () {
    const { registry, publisher1, publisher2 } = await deployFixture();
    await registry.connect(publisher1).registerDocument(HASH_1, "cid-1");
    await expect(
      registry.connect(publisher2).registerDocument(HASH_1, "cid-2")
    ).to.be.reverted;
  });

  it("auto-authorizes the deployer as an oracle at construction", async function () {
    const { registry, owner } = await deployFixture();
    expect(await registry.authorizedOracles(owner.address)).to.equal(true);
  });

  it("reverts when a non-owner calls authorizeOracle", async function () {
    const { registry, publisher1, oracle2 } = await deployFixture();
    await expect(
      registry.connect(publisher1).authorizeOracle(oracle2.address, true)
    ).to.be.reverted;
  });

  it("allows the owner to authorize a new oracle", async function () {
    const { registry, owner, oracle2 } = await deployFixture();
    await registry.connect(owner).authorizeOracle(oracle2.address, true);
    expect(await registry.authorizedOracles(oracle2.address)).to.equal(true);
  });

  it("reverts when a non-authorized address calls updateReputation", async function () {
    const { registry, publisher1 } = await deployFixture();
    await registry.connect(publisher1).registerDocument(HASH_1, "cid-1");
    await expect(
      registry.connect(publisher1).updateReputation(HASH_1, -5)
    ).to.be.reverted;
  });

  it("allows an authorized oracle to adjust reputation, including into negative territory", async function () {
    const { registry, owner, publisher1 } = await deployFixture();
    await registry.connect(publisher1).registerDocument(HASH_1, "cid-1");
    await registry.connect(owner).updateReputation(HASH_1, -10);
    const doc = await registry.documentLedger(HASH_1);
    expect(doc.reputationScore).to.equal(-10);
    await registry.connect(owner).updateReputation(HASH_1, -5);
    const doc2 = await registry.documentLedger(HASH_1);
    expect(doc2.reputationScore).to.equal(-15);
  });

  it("reverts updateReputation for an unregistered hash even from an authorized oracle", async function () {
    const { registry, owner } = await deployFixture();
    await expect(
      registry.connect(owner).updateReputation(HASH_UNREGISTERED, 5)
    ).to.be.reverted;
  });

  it("verifyIntegrity returns false/0/zero-address for an unregistered hash", async function () {
    const { registry } = await deployFixture();
    const [exists, reputation, publisher] = await registry.verifyIntegrity(HASH_UNREGISTERED);
    expect(exists).to.equal(false);
    expect(reputation).to.equal(0);
    expect(publisher).to.equal(ethers.ZeroAddress);
  });

  it("verifyIntegrity returns correct values for a registered hash", async function () {
    const { registry, publisher1 } = await deployFixture();
    await registry.connect(publisher1).registerDocument(HASH_1, "cid-1");
    const [exists, reputation, publisher] = await registry.verifyIntegrity(HASH_1);
    expect(exists).to.equal(true);
    expect(reputation).to.equal(0);
    expect(publisher).to.equal(publisher1.address);
  });

  it("documentLedger getter returns all six fields for a registered document", async function () {
    const { registry, publisher1 } = await deployFixture();
    await registry.connect(publisher1).registerDocument(HASH_1, "cid-1");
    const doc = await registry.documentLedger(HASH_1);
    expect(doc.publisher).to.equal(publisher1.address);
    expect(doc.contentHash).to.equal(HASH_1);
    expect(doc.ipfsCID).to.equal("cid-1");
    expect(doc.reputationScore).to.equal(0);
    expect(doc.timestamp).to.be.greaterThan(0);
    expect(doc.isRegistered).to.equal(true);
  });
});
