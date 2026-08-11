/**
 * Deploy the ADAM governance contracts and verify on-chain state against the
 * manuscript.
 *
 * The post-deploy assertions matter: a contract that deploys but reports a
 * quorum disagreeing with Table 8, or a screening threshold that is not 2% of
 * the LEL, is a contract that will embarrass the paper. This script refuses to
 * report success in that case.
 *
 *   npx hardhat run scripts/deploy.js --network fides
 */
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

// Table 8 of the manuscript.
const TABLE_8_QUORUM = { 2: 2, 3: 3, 4: 3, 5: 4, 6: 4, 7: 5 };
const TABLE_8_FAULTS = { 2: 0, 3: 0, 4: 1, 5: 1, 6: 2, 7: 2 };

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  const balance = await hre.ethers.provider.getBalance(deployer.address);
  console.log(`Deployer: ${deployer.address}`);
  console.log(`Balance:  ${hre.ethers.formatEther(balance)}\n`);

  const GovernanceRules = await hre.ethers.getContractFactory("GovernanceRules");
  const governance = await GovernanceRules.deploy();
  await governance.waitForDeployment();
  const governanceAddress = await governance.getAddress();
  console.log(`GovernanceRules     ${governanceAddress}`);

  const CrewRegistry = await hre.ethers.getContractFactory("CrewRegistry");
  const registry = await CrewRegistry.deploy();
  await registry.waitForDeployment();
  const registryAddress = await registry.getAddress();
  console.log(`CrewRegistry        ${registryAddress}`);

  const DecisionLogger = await hre.ethers.getContractFactory("DecisionLogger");
  const logger = await DecisionLogger.deploy();
  await logger.waitForDeployment();
  const loggerAddress = await logger.getAddress();
  console.log(`DecisionLogger      ${loggerAddress}`);

  const ConsensusValidator = await hre.ethers.getContractFactory("ConsensusValidator");
  const validator = await ConsensusValidator.deploy(governanceAddress, registryAddress);
  await validator.waitForDeployment();
  const validatorAddress = await validator.getAddress();
  console.log(`ConsensusValidator  ${validatorAddress}\n`);

  // Distinct addresses. A duplicate here means two names point at one
  // deployment, which invalidates any per-contract cost or trace attribution.
  const addresses = {
    GovernanceRules: governanceAddress,
    CrewRegistry: registryAddress,
    DecisionLogger: loggerAddress,
    ConsensusValidator: validatorAddress,
  };
  const seen = new Set();
  for (const [name, addr] of Object.entries(addresses)) {
    if (seen.has(addr.toLowerCase())) {
      throw new Error(`Duplicate deployed address for ${name}: ${addr}`);
    }
    seen.add(addr.toLowerCase());
  }

  // -- verify against the manuscript
  console.log("Verifying on-chain state against the manuscript:");
  const screening = await governance.screeningThreshold();
  const pctScaled = await governance.thresholdPercentOfLelScaled();
  console.log(`  screeningThreshold      ${screening} ppm`);
  console.log(`  as % of LEL             ${Number(pctScaled) / 100}%`);
  if (Number(screening) !== 1000) {
    throw new Error(`screeningThreshold is ${screening}, constraint C5 requires 1000`);
  }
  if (Number(pctScaled) !== 200) {
    throw new Error(`threshold is ${Number(pctScaled) / 100}% of LEL, C1 argument requires 2%`);
  }

  for (const [n, expected] of Object.entries(TABLE_8_QUORUM)) {
    const got = Number(await governance.requiredQuorum(n));
    const faults = Number(await governance.toleratedFaults(n));
    const okQ = got === expected;
    const okF = faults === TABLE_8_FAULTS[n];
    console.log(
      `  n=${n}  quorum ${got} (Table 8: ${expected}) ${okQ ? "ok" : "MISMATCH"}` +
        `   f=${faults} (${TABLE_8_FAULTS[n]}) ${okF ? "ok" : "MISMATCH"}`
    );
    if (!okQ || !okF) {
      throw new Error(`Contract disagrees with Table 8 at crew size ${n}`);
    }
  }

  const deployment = {
    network: hre.network.name,
    chainId: Number((await hre.ethers.provider.getNetwork()).chainId),
    deployer: deployer.address,
    timestamp: new Date().toISOString(),
    contracts: addresses,
    verifiedAgainstManuscript: {
      screeningThresholdPpm: Number(screening),
      thresholdPercentOfLel: Number(pctScaled) / 100,
      quorumRule: "ceil(n/2) + 1  (Equation 4)",
      table8Verified: true,
    },
  };

  const outDir = path.join(__dirname, "..", "blockchain");
  fs.mkdirSync(outDir, { recursive: true });
  const outPath = path.join(outDir, `deployment.${hre.network.name}.json`);
  fs.writeFileSync(outPath, JSON.stringify(deployment, null, 2));

  console.log(`\nAll checks passed. Wrote ${outPath}`);
  console.log("\nExport for the Python runtime:");
  for (const [name, addr] of Object.entries(addresses)) {
    const envName = name.replace(/([a-z])([A-Z])/g, "$1_$2").toUpperCase();
    console.log(`  export ADAM_ADDR_${envName}=${addr}`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
