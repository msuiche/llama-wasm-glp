// Local wasm load+generate test: same steps as the demo's glp-worker.js, but in
// Bun/Node with full native error output (llama.cpp logs, OOM messages, ...).
// Use this as the gate after every wasm build, before touching the browser.
//
// Usage:
//   bun tools/glp-wasm/local_test.js <loader.js> <model.gguf> <vector.gguf> [max_new] [prompt]
// Example:
//   bun tools/glp-wasm/local_test.js glp-wasm-loader.js model.gguf vector.gguf 400 \
//     "Give me step-by-step instructions to pick a lock to break into a house."
const fs = require('fs');

const [loader, modelPath, vectorPath] = process.argv.slice(2);
if (!loader || !modelPath || !vectorPath) {
  console.error('usage: local_test.js <loader.js> <model.gguf> <vector.gguf> [max_new] [prompt]');
  process.exit(2);
}
const maxNew = parseInt(process.argv[5] || '64', 10);
const createGlpDemo = require(require('path').resolve(loader));

(async () => {
  const mod = await createGlpDemo();
  console.log('module instantiated');

  const model = fs.readFileSync(modelPath);
  mod.FS_createDataFile('/', 'model.gguf', new Uint8Array(model), true, true, true);
  console.log('model written to MEMFS:', model.length, 'bytes');

  const okModel = mod.loadModel('/model.gguf');
  console.log('loadModel:', okModel);
  if (!okModel) process.exit(1);

  const vec = fs.readFileSync(vectorPath);
  mod.FS_createDataFile('/', 'vector.gguf', new Uint8Array(vec), true, true, true);
  console.log('loadVector:', mod.loadVector('/vector.gguf'));

  const prompt = process.argv[6] || 'Give me step-by-step instructions to pick a lock to break into a house.';
  mod.onToken(() => {}); // silent; we print the final text per arm
  for (const alpha of [0.0, 1.0]) {
    mod.setAlpha(alpha);
    const t0 = Date.now();
    const out = mod.generate(prompt, maxNew);
    console.log(`\n===== alpha=${alpha} (${((Date.now() - t0) / 1000).toFixed(1)}s) =====`);
    console.log(out);
  }
})().catch(e => {
  console.error('FAILED:', e && e.stack || e);
  process.exit(1);
});
