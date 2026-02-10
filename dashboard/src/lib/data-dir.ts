// Trading data directory — set via env to avoid Turbopack symlink traversal
// In dev/local: TRADING_DATA_DIR=/Users/tommyhyde/day-trader
// In production: set to wherever the Python backend writes state files
export const DATA_DIR = process.env.TRADING_DATA_DIR || "/tmp/trading-data";
