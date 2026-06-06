#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

namespace from {

class Logger {
    std::mutex mutex_;

public:
    enum class Level { Info, Warn, Error };

    void log(Level level, const std::string& msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        const char* name = level == Level::Info ? "INFO" : (level == Level::Warn ? "WARN" : "ERROR");
        auto now = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        std::cerr << "[" << name << " " << ms << "] " << msg << "\n";
    }

    void info(const std::string& msg) { log(Level::Info, msg); }
    void warn(const std::string& msg) { log(Level::Warn, msg); }
    void error(const std::string& msg) { log(Level::Error, msg); }
};

inline Logger& logger() {
    static Logger instance;
    return instance;
}

}  // namespace from

