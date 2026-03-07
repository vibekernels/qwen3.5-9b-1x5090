#include "download.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <pwd.h>
#include <thread>
#include <atomic>

// We use nlohmann::json for parsing the HF API response
#include "json.hpp"
using json = nlohmann::json;

std::string default_model_dir() {
    const char* home = getenv("HOME");
    if (!home) {
        struct passwd* pw = getpwuid(getuid());
        if (pw) home = pw->pw_dir;
    }
    if (!home) home = "/tmp";
    return std::string(home) + "/.cache/qwen-models";
}

static bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

static int64_t file_size(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) == 0) return st.st_size;
    return -1;
}

static void mkdirs(const std::string& path) {
    // Create directory tree recursively
    std::string accum;
    for (size_t i = 0; i < path.size(); i++) {
        accum += path[i];
        if (path[i] == '/' || i == path.size() - 1) {
            mkdir(accum.c_str(), 0755);
        }
    }
}

// Run a command and capture stdout
static std::string capture_cmd(const std::string& cmd) {
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "";
    std::string result;
    char buf[4096];
    while (fgets(buf, sizeof(buf), pipe)) {
        result += buf;
    }
    pclose(pipe);
    return result;
}

// Parse HF spec: "org/repo:filter" -> (org/repo, filter)
// Also handles "org/repo" with no filter
static bool parse_hf_spec(const std::string& spec, std::string& repo, std::string& filter) {
    auto colon = spec.find(':');
    if (colon != std::string::npos) {
        repo = spec.substr(0, colon);
        filter = spec.substr(colon + 1);
    } else {
        repo = spec;
        filter = "";
    }
    // Validate: must have exactly one '/'
    auto slash = repo.find('/');
    if (slash == std::string::npos || slash == 0 || slash == repo.size() - 1) return false;
    if (repo.find('/', slash + 1) != std::string::npos) return false;
    return true;
}

// Case-insensitive substring check
static bool icontains(const std::string& haystack, const std::string& needle) {
    if (needle.empty()) return true;
    auto it = std::search(haystack.begin(), haystack.end(), needle.begin(), needle.end(),
        [](char a, char b) { return tolower(a) == tolower(b); });
    return it != haystack.end();
}

// Find matching GGUF file in HF repo using API
static std::string find_hf_gguf(const std::string& repo, const std::string& filter) {
    // Query HF API for file listing
    std::string cmd = "curl -sL 'https://huggingface.co/api/models/" + repo + "'";
    std::string response = capture_cmd(cmd);
    if (response.empty()) {
        fprintf(stderr, "Error: failed to query HuggingFace API for %s\n", repo.c_str());
        return "";
    }

    try {
        json j = json::parse(response);
        if (!j.contains("siblings")) {
            if (j.contains("error")) {
                fprintf(stderr, "Error: HuggingFace API: %s\n", j["error"].get<std::string>().c_str());
            } else {
                fprintf(stderr, "Error: unexpected HuggingFace API response\n");
            }
            return "";
        }

        std::vector<std::string> gguf_files;
        for (auto& sibling : j["siblings"]) {
            std::string fname = sibling["rfilename"].get<std::string>();
            if (fname.size() > 5 && fname.substr(fname.size() - 5) == ".gguf") {
                gguf_files.push_back(fname);
            }
        }

        if (gguf_files.empty()) {
            fprintf(stderr, "Error: no .gguf files found in %s\n", repo.c_str());
            return "";
        }

        // If no filter, return the first (or only) GGUF file
        if (filter.empty()) {
            if (gguf_files.size() == 1) return gguf_files[0];
            fprintf(stderr, "Error: multiple .gguf files in %s, specify one with :filter\n", repo.c_str());
            fprintf(stderr, "Available files:\n");
            for (auto& f : gguf_files) fprintf(stderr, "  %s\n", f.c_str());
            return "";
        }

        // Match filter against filenames
        std::vector<std::string> matches;
        for (auto& f : gguf_files) {
            if (icontains(f, filter)) matches.push_back(f);
        }

        if (matches.empty()) {
            fprintf(stderr, "Error: no .gguf file matching '%s' in %s\n", filter.c_str(), repo.c_str());
            fprintf(stderr, "Available files:\n");
            for (auto& f : gguf_files) fprintf(stderr, "  %s\n", f.c_str());
            return "";
        }
        if (matches.size() > 1) {
            // Prefer exact case-insensitive match on the quant part
            for (auto& f : matches) {
                // Extract quant: typically "Model-QUANT.gguf"
                auto dash = f.rfind('-');
                auto dot = f.rfind('.');
                if (dash != std::string::npos && dot != std::string::npos && dot > dash) {
                    std::string quant = f.substr(dash + 1, dot - dash - 1);
                    if (quant.size() == filter.size() && icontains(quant, filter)) {
                        return f;
                    }
                }
            }
            // Just take the first match
        }
        return matches[0];
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: failed to parse HuggingFace API response: %s\n", e.what());
        return "";
    }
}


std::string resolve_model(const std::string& spec, const std::string& model_dir_arg,
                          DownloadCallback callback) {
    // If spec points to an existing local file, use it directly
    if (file_exists(spec)) return spec;

    // Parse as HF spec
    std::string repo, filter;
    if (!parse_hf_spec(spec, repo, filter)) {
        fprintf(stderr, "Error: '%s' is not a valid file path or HuggingFace tag (org/repo:filter)\n", spec.c_str());
        return "";
    }

    std::string model_dir = model_dir_arg.empty() ? default_model_dir() : model_dir_arg;

    // Convert repo to directory: "org/repo" -> "model_dir/org/repo"
    std::string cache_dir = model_dir + "/" + repo;
    mkdirs(cache_dir);

    // Check if we already have a matching GGUF file cached
    // First, try to find the exact file from HF without querying the API
    // by scanning the cache directory
    DIR* d = opendir(cache_dir.c_str());
    if (d) {
        struct dirent* entry;
        while ((entry = readdir(d)) != nullptr) {
            std::string fname = entry->d_name;
            if (fname.size() > 5 && fname.substr(fname.size() - 5) == ".gguf") {
                if (filter.empty() || icontains(fname, filter)) {
                    closedir(d);
                    std::string path = cache_dir + "/" + fname;
                    fprintf(stderr, "Using cached model: %s\n", path.c_str());
                    return path;
                }
            }
        }
        closedir(d);
    }

    // Not cached — query HF API to find the right file
    fprintf(stderr, "Resolving %s from HuggingFace...\n", spec.c_str());
    std::string gguf_filename = find_hf_gguf(repo, filter);
    if (gguf_filename.empty()) return "";

    std::string local_path = cache_dir + "/" + gguf_filename;
    std::string url = "https://huggingface.co/" + repo + "/resolve/main/" + gguf_filename;

    fprintf(stderr, "Downloading %s\n  -> %s\n", url.c_str(), local_path.c_str());

    // Download with progress tracking
    // Use a .partial suffix during download to avoid serving incomplete files
    std::string partial_path = local_path + ".partial";

    if (callback) {
        // Get file size first via HEAD request
        std::string head_cmd = "curl -sLI '" + url + "' 2>/dev/null | grep -i content-length | tail -1 | awk '{print $2}' | tr -d '\\r'";
        std::string size_str = capture_cmd(head_cmd);
        int64_t total_size = 0;
        if (!size_str.empty()) total_size = atoll(size_str.c_str());

        // Run curl in a separate thread so we can poll file size for progress
        std::atomic<bool> dl_done{false};
        std::atomic<int> dl_exit{0};
        std::string dl_cmd = "curl -sL -o '" + partial_path + "' '" + url + "'";
        std::thread dl_thread([&dl_done, &dl_exit, dl_cmd]() {
            dl_exit.store(system(dl_cmd.c_str()));
            dl_done.store(true);
        });

        // Poll file size for progress until curl finishes
        while (!dl_done.load()) {
            int64_t current = file_size(partial_path);
            if (current < 0) current = 0;
            callback(current, total_size);
            usleep(500000);  // 500ms
        }
        dl_thread.join();

        int64_t final_dl = file_size(partial_path);
        if (final_dl > 0) callback(final_dl, total_size > 0 ? total_size : final_dl);

        if (dl_exit.load() != 0) {
            fprintf(stderr, "Error: download failed (curl exit code %d)\n", dl_exit.load());
            unlink(partial_path.c_str());
            return "";
        }
    } else {
        // No callback — show curl's built-in progress bar
        std::string dl_cmd = "curl -L --progress-bar -o '" + partial_path + "' '" + url + "'";
        int ret = system(dl_cmd.c_str());
        if (ret != 0) {
            fprintf(stderr, "Error: download failed (curl exit code %d)\n", ret);
            unlink(partial_path.c_str());
            return "";
        }
    }

    // Verify download
    int64_t final_size = file_size(partial_path);
    if (final_size <= 0) {
        fprintf(stderr, "Error: download produced empty file\n");
        unlink(partial_path.c_str());
        return "";
    }

    // Rename partial to final
    if (rename(partial_path.c_str(), local_path.c_str()) != 0) {
        fprintf(stderr, "Error: failed to rename %s -> %s\n", partial_path.c_str(), local_path.c_str());
        return "";
    }

    fprintf(stderr, "Download complete: %s (%.1f GB)\n", local_path.c_str(), final_size / 1e9);
    return local_path;
}
