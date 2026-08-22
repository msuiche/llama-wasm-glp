// Read a real control vector through llama.cpp's loader and print what it found.
//
// Exists to cross-check one artifact against two independent readers: the vLLM
// overlay (Python, gguf-py) and this one (C++, gguf.c). Both must agree on mode,
// hook point, alpha, layer set and the 1-based offset -- a file that loads in one
// and shifts by a layer in the other is worse than a file that fails outright.
// No model needed, so it runs against a 300B-class model's vector on a laptop.

#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) { printf("usage: %s <cvec.gguf> [strength]\n", argv[0]); return 1; }
    const float strength = argc > 2 ? std::atof(argv[2]) : 1.0f;

    std::vector<common_control_vector_load_info> in;
    in.push_back({strength, argv[1]});
    const auto cv = common_control_vector_load(in);

    if (cv.n_embd == -1) { printf("  REFUSED by loader\n"); return 1; }

    const size_t n_layers = cv.data.size() / cv.n_embd;
    printf("  n_embd           %d\n", cv.n_embd);
    printf("  layer slots      %zu  (direction.1 .. direction.%zu)\n", n_layers, n_layers);
    printf("  mode             %s\n", cv.mode == LLAMA_CVEC_APPLY_PROJECT ? "project" : "add");
    printf("  alpha            %.3f\n", cv.alpha);

    // Which slots actually carry a direction, and is each one unit-norm? A
    // populated slot at the wrong index is the failure mode the 1-based
    // convention invites, and it is silent.
    int n_pop = 0, n_unit = 0, first = -1, last = -1;
    double min_n = 1e30, max_n = 0.0;
    std::string ids;
    for (size_t il = 0; il < n_layers; il++) {
        double sq = 0.0;
        for (int j = 0; j < cv.n_embd; j++) {
            const float v = cv.data[il*cv.n_embd + j];
            sq += (double) v*v;
        }
        if (sq <= 1e-12) { continue; }
        const double nrm = std::sqrt(sq);
        n_pop++;
        if (std::fabs(nrm - 1.0) < 1e-3) { n_unit++; }
        min_n = std::min(min_n, nrm);
        max_n = std::max(max_n, nrm);
        // data[il] holds direction.(il+1), which is model layer (il+1)
        const int layer = (int) il + 1;
        if (first < 0) { first = layer; }
        last = layer;
        ids += (ids.empty() ? "" : ",") + std::to_string(layer);
    }
    printf("  populated        %d of %zu slots\n", n_pop, n_layers);
    printf("  model layers     %d..%d\n", first, last);
    printf("  layer ids        %s\n", ids.c_str());
    printf("  norms            %.6f .. %.6f  (%d of %d unit)\n", min_n, max_n, n_unit, n_pop);

    const bool ok = n_pop > 0 && n_unit == n_pop;
    printf(ok ? "  inspect: OK\n" : "  inspect: NON-UNIT DIRECTIONS PRESENT\n");
    return ok ? 0 : 1;
}
