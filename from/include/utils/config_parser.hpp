#pragma once

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace from {

class Config {
    std::unordered_map<std::string, std::string> values_;

    static std::string trim(std::string s) {
        auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
        s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
        return s;
    }

    static std::string strip_quotes(std::string s) {
        s = trim(std::move(s));
        if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') || (s.front() == '\'' && s.back() == '\''))) {
            return s.substr(1, s.size() - 2);
        }
        return s;
    }

public:
    void load(const std::string& path) {
        std::ifstream in(path);
        if (!in) {
            throw std::runtime_error("Cannot open config: " + path);
        }
        std::string section;
        std::string line;
        while (std::getline(in, line)) {
            size_t comment = line.find('#');
            if (comment != std::string::npos) {
                line = line.substr(0, comment);
            }
            line = trim(line);
            if (line.empty()) {
                continue;
            }
            if (line.front() == '[' && line.back() == ']') {
                section = trim(line.substr(1, line.size() - 2));
                continue;
            }
            size_t eq = line.find('=');
            if (eq == std::string::npos) {
                continue;
            }
            std::string key = trim(line.substr(0, eq));
            std::string val = trim(line.substr(eq + 1));
            values_[section.empty() ? key : section + "." + key] = strip_quotes(val);
        }
    }

    bool has(const std::string& key) const { return values_.find(key) != values_.end(); }

    std::string get_string(const std::string& key, const std::string& def = "") const {
        auto it = values_.find(key);
        return it == values_.end() ? def : strip_quotes(it->second);
    }

    int get_int(const std::string& key, int def = 0) const {
        auto it = values_.find(key);
        return it == values_.end() ? def : std::stoi(it->second);
    }

    size_t get_size(const std::string& key, size_t def = 0) const {
        auto it = values_.find(key);
        return it == values_.end() ? def : static_cast<size_t>(std::stoull(it->second));
    }

    float get_float(const std::string& key, float def = 0.0f) const {
        auto it = values_.find(key);
        return it == values_.end() ? def : std::stof(it->second);
    }

    bool get_bool(const std::string& key, bool def = false) const {
        auto it = values_.find(key);
        if (it == values_.end()) {
            return def;
        }
        std::string v = it->second;
        std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return v == "true" || v == "1" || v == "yes";
    }

    std::vector<float> get_float_list(const std::string& key, const std::vector<float>& def = {}) const {
        auto it = values_.find(key);
        if (it == values_.end()) {
            return def;
        }
        std::string s = it->second;
        if (!s.empty() && s.front() == '[') {
            s.erase(s.begin());
        }
        if (!s.empty() && s.back() == ']') {
            s.pop_back();
        }
        std::vector<float> out;
        std::stringstream ss(s);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            tok = trim(tok);
            if (!tok.empty()) {
                out.push_back(std::stof(tok));
            }
        }
        return out;
    }
};

}  // namespace from
