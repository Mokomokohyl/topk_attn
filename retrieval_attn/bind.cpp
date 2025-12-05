#include <torch/extension.h>
#include <cuda_fp16.h>

// Declarations
void launch_retrieval_attention_32(
    const half* query,
    const half* key_cache,
    const half* value_cache,
    half* output
);

void launch_retrieval_attention_128(
    const half* query,
    const half* key_cache,
    const half* value_cache,
    half* output
);

void launch_retrieval_attention_256(
    const half* query,
    const half* key_cache,
    const half* value_cache,
    half* output
);

void launch_retrieval_attention_512(
    const half* query,
    const half* key_cache,
    const half* value_cache,
    half* output
);

void launch_retrieval_attention_1024(
    const half* query,
    const half* key_cache,
    const half* value_cache,
    half* output
);

void launch_retrieval_attention_global_32(
    const half* query,
    const half* key_cache,
    const half* value_cache,
    half* output
);

void launch_retrieval_attention_global_128(
    const half* query,
    const half* key_cache,
    const half* value_cache,
    half* output
);

void launch_retrieval_attention_global_256(
    const half* query,
    const half* key_cache,
    const half* value_cache,
    half* output
);

void launch_retrieval_attention_global_512(
    const half* query,
    const half* key_cache,
    const half* value_cache,
    half* output
);

void launch_retrieval_attention_global_1024(
    const half* query,
    const half* key_cache,
    const half* value_cache,
    half* output
);

// PyTorch wrappers
void retrieval_attention_32(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor output
) {
    launch_retrieval_attention_32(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>())
    );
}

void retrieval_attention_128(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor output
) {
    launch_retrieval_attention_128(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>())
    );
}

void retrieval_attention_256(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor output
) {
    launch_retrieval_attention_256(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>())
    );
}

void retrieval_attention_512(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor output
) {
    launch_retrieval_attention_512(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>())
    );
}

void retrieval_attention_1024(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor output
) {
    launch_retrieval_attention_1024(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>())
    );
}

void retrieval_attention_global_32(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor output
) {
    launch_retrieval_attention_global_32(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>())
    );
}

void retrieval_attention_global_128(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor output
) {
    launch_retrieval_attention_global_128(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>())
    );
}

void retrieval_attention_global_256(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor output
) {
    launch_retrieval_attention_global_256(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>())
    );
}

void retrieval_attention_global_512(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor output
) {
    launch_retrieval_attention_global_512(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>())
    );
}

void retrieval_attention_global_1024(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor output
) {
    launch_retrieval_attention_global_1024(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>())
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("retrieval_attention_32", &retrieval_attention_32, "Retrieval Attention (TopK=32)");
    m.def("retrieval_attention_128", &retrieval_attention_128, "Retrieval Attention (TopK=128)");
    m.def("retrieval_attention_256", &retrieval_attention_256, "Retrieval Attention (TopK=256)");
    m.def("retrieval_attention_512", &retrieval_attention_512, "Retrieval Attention (TopK=512)");
    m.def("retrieval_attention_1024", &retrieval_attention_1024, "Retrieval Attention (TopK=1024)");
    m.def("retrieval_attention_global_32", &retrieval_attention_global_32, "Retrieval Attention Global (TopK=32)");
    m.def("retrieval_attention_global_128", &retrieval_attention_global_128, "Retrieval Attention Global (TopK=128)");
    m.def("retrieval_attention_global_256", &retrieval_attention_global_256, "Retrieval Attention Global (TopK=256)");
    m.def("retrieval_attention_global_512", &retrieval_attention_global_512, "Retrieval Attention Global (TopK=512)");
    m.def("retrieval_attention_global_1024", &retrieval_attention_global_1024, "Retrieval Attention Global (TopK=1024)");
}
