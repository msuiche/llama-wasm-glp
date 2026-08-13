// Numerical check of the projective control vector apply.
//
// llama_adapter_cvec::apply_to() composes mul_mat/repeat/mul/scale/sub to get
// h -= alpha*(h.d)d per token. That composition relies on ggml broadcasting
// rules that are easy to get subtly wrong -- a transposed operand or a broadcast
// along the wrong axis still produces a same-shaped tensor and plausible output.
// So build the same graph here on known values and compare against a hand
// computed projection.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int n_fail = 0;

static void check(const char * what, float got, float want, float tol = 1e-4f) {
    const bool ok = std::fabs(got - want) <= tol;
    if (!ok) {
        printf("  FAIL %-28s got %+.6f want %+.6f\n", what, got, want);
        n_fail++;
    }
}

// same op sequence as llama_adapter_cvec::apply_to() in project mode
static std::vector<float> apply_project(
        const std::vector<float> & h_in,  // [n_embd * n_tokens], column-major per token
        const std::vector<float> & d_in,  // [n_embd]
        int n_embd, int n_tokens, float alpha) {
    ggml_init_params ip = { 256*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * d = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    memcpy(h->data, h_in.data(), h_in.size()*sizeof(float));
    memcpy(d->data, d_in.data(), d_in.size()*sizeof(float));

    ggml_tensor * coef = ggml_mul_mat(ctx, d, h);
    ggml_tensor * proj = ggml_mul(ctx, ggml_repeat(ctx, d, h), coef);
    ggml_tensor * out  = ggml_sub(ctx, h, ggml_scale(ctx, proj, alpha));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    std::vector<float> res((size_t) n_embd*n_tokens);
    memcpy(res.data(), out->data, res.size()*sizeof(float));
    ggml_free(ctx);
    return res;
}

int main() {
    const int n_embd = 8;
    const int n_tokens = 5;

    // unit direction, deliberately not axis-aligned
    std::vector<float> d(n_embd);
    float nrm = 0.0f;
    for (int i = 0; i < n_embd; i++) { d[i] = std::sin(0.7f*i + 0.3f); nrm += d[i]*d[i]; }
    nrm = std::sqrt(nrm);
    for (int i = 0; i < n_embd; i++) { d[i] /= nrm; }

    std::vector<float> h((size_t) n_embd*n_tokens);
    for (int t = 0; t < n_tokens; t++) {
        for (int i = 0; i < n_embd; i++) {
            h[(size_t) t*n_embd + i] = std::cos(1.1f*i + 0.5f*t) * (1.0f + 0.3f*t);
        }
    }

    printf("test-cvec-project: n_embd=%d n_tokens=%d\n", n_embd, n_tokens);

    // 1. alpha = 1 removes the component exactly: (h' . d) == 0, per token.
    {
        auto out = apply_project(h, d, n_embd, n_tokens, 1.0f);
        for (int t = 0; t < n_tokens; t++) {
            double dot = 0.0, dot_in = 0.0;
            for (int i = 0; i < n_embd; i++) {
                dot    += (double) out[(size_t) t*n_embd + i] * d[i];
                dot_in += (double) h  [(size_t) t*n_embd + i] * d[i];
            }
            char buf[64]; snprintf(buf, sizeof(buf), "alpha=1 residual dot t%d", t);
            check(buf, (float) dot, 0.0f);
            // and the input was not already orthogonal, or the test proves nothing
            if (std::fabs(dot_in) < 1e-3) { printf("  FAIL degenerate input t%d\n", t); n_fail++; }
        }
    }

    // 2. elementwise against a hand computed projection at a non-unit alpha.
    //    This is what catches a broadcast along the wrong axis: test 1 passes for
    //    some wrong compositions, this does not.
    {
        const float alpha = 4.0f;
        auto out = apply_project(h, d, n_embd, n_tokens, alpha);
        for (int t = 0; t < n_tokens; t++) {
            double dot = 0.0;
            for (int i = 0; i < n_embd; i++) { dot += (double) h[(size_t) t*n_embd + i] * d[i]; }
            for (int i = 0; i < n_embd; i++) {
                const float want = h[(size_t) t*n_embd + i] - alpha * (float) dot * d[i];
                char buf[64]; snprintf(buf, sizeof(buf), "alpha=4 elem t%d i%d", t, i);
                check(buf, out[(size_t) t*n_embd + i], want);
            }
        }
    }

    // 3. alpha = 0 is exact identity. The vLLM port shipped a bug where the
    //    alpha=0 control traced a graph with the op removed and cached it, so the
    //    "no-op" arm was not measuring the same graph. Assert bit-level equality.
    {
        auto out = apply_project(h, d, n_embd, n_tokens, 0.0f);
        for (size_t i = 0; i < h.size(); i++) {
            if (out[i] != h[i]) { printf("  FAIL alpha=0 not identity at %zu\n", i); n_fail++; break; }
        }
    }

    // 4. a token already orthogonal to d must be untouched at any alpha.
    {
        std::vector<float> h2((size_t) n_embd, 0.0f);
        // Gram-Schmidt one vector against d
        double dot = 0.0;
        for (int i = 0; i < n_embd; i++) { h2[i] = std::cos(2.3f*i); dot += (double) h2[i]*d[i]; }
        for (int i = 0; i < n_embd; i++) { h2[i] -= (float) dot * d[i]; }
        auto out = apply_project(h2, d, n_embd, 1, 4.0f);
        for (int i = 0; i < n_embd; i++) {
            check("orthogonal token unchanged", out[i], h2[i]);
        }
    }

    // 5. projection is idempotent at alpha=1: applying twice equals applying once.
    {
        auto once  = apply_project(h, d, n_embd, n_tokens, 1.0f);
        auto twice = apply_project(once, d, n_embd, n_tokens, 1.0f);
        for (size_t i = 0; i < once.size(); i++) {
            check("idempotent", twice[i], once[i]);
        }
    }

    printf(n_fail == 0 ? "test-cvec-project: OK\n" : "test-cvec-project: %d FAILURES\n", n_fail);
    return n_fail == 0 ? 0 : 1;
}
