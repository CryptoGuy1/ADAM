require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

/**
 * Hardhat configuration for the ADAM governance contracts.
 *
 * The `fides` network targets the Fides Innova PoA testnet used in the
 * manuscript (Section 3.4.1). `localhost` runs the same contracts against a
 * local node so the test suite and the offline harness need no testnet funds.
 */
module.exports = {
  solidity: {
    version: "0.8.24",
    settings: {
      optimizer: { enabled: true, runs: 200 },
    },
  },
  paths: {
    sources: "./contracts",
    artifacts: "./blockchain/artifacts",
    cache: "./blockchain/cache",
  },
  networks: {
    hardhat: { chainId: 31337 },
    localhost: { url: "http://127.0.0.1:8545", chainId: 31337 },
    fides: {
      url: process.env.ADAM_CHAIN_RPC || "https://fidesf1-rpc.fidesinnova.io",
      chainId: parseInt(process.env.ADAM_CHAIN_ID || "706883", 10),
      accounts: process.env.DEPLOYER_PRIVATE_KEY
        ? [process.env.DEPLOYER_PRIVATE_KEY]
        : [],
    },
  },
};
