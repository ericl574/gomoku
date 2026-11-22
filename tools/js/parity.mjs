// Run site/engine.js's MCTS on the positions in a fixture and print the visit counts as JSON.
// Driven by check_site_engine.py, which runs the Python search on the same positions and compares.
//
//   node tools/js/parity.mjs <fixture.json>
//
// fixture: {"model": "site/model.onnx" | "flat", "simulations": N, "positions": [[action, ...], ...]}

import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { GomokuState, MCTS, CELLS } = require("../../site/engine.js");

const fixture = JSON.parse(readFileSync(process.argv[2], "utf8"));

let evaluate;
if (fixture.model === "flat") {
  // Same as test_alphazero.FlatNetwork: equal logits everywhere, value 0.
  evaluate = async () => ({ logits: new Float32Array(CELLS), value: 0 });
} else {
  const ort = require("onnxruntime-node");
  const session = await ort.InferenceSession.create(fixture.model);
  evaluate = async (planes) => {
    const out = await session.run({ board: new ort.Tensor("float32", planes, [1, 3, 19, 19]) });
    return { logits: out.policy_logits.data, value: out.value.data[0] };
  };
}

const results = [];
for (const moves of fixture.positions) {
  const state = new GomokuState();
  for (const action of moves) state.step(action);
  const search = new MCTS(evaluate, { simulations: fixture.simulations, yieldEvery: 0 });
  const { root } = await search.search(state);
  const visits = {};
  for (const [action, child] of root.children) if (child.visitCount) visits[action] = child.visitCount;
  results.push({ visits, stats: search.lastSearchStats });
}
process.stdout.write(JSON.stringify(results));
