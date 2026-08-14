import { expect } from "chai";
import { ethers } from "hardhat";

describe("Placeholder", function () {
  it("stores and returns a value", async function () {
    const Placeholder = await ethers.getContractFactory("Placeholder");
    const placeholder = await Placeholder.deploy();
    await placeholder.setValue(42);
    expect(await placeholder.getValue()).to.equal(42);
  });
});
