#pragma once

#include <string>
#include <functional>
#include <atomic>

struct DownloadProgress {
    std::atomic<int64_t> downloaded{0};
    std::atomic<int64_t> total{0};
    std::atomic<bool> complete{false};
    std::atomic<bool> failed{false};
    std::string error;
    std::string filename;
};

// Progress callback: (bytes_downloaded, bytes_total)
using DownloadCallback = std::function<void(int64_t, int64_t)>;

// Resolve a model spec to a local file path, downloading from HuggingFace if needed.
// spec: "org/repo:filter" (HF tag) or local file path
// model_dir: cache directory (default: ~/.cache/qwen-models)
// callback: optional progress callback during download
// Returns: local path to the .gguf file, or empty string on failure.
std::string resolve_model(const std::string& spec, const std::string& model_dir = "",
                          DownloadCallback callback = nullptr);

// Default model directory: ~/.cache/qwen-models
std::string default_model_dir();
