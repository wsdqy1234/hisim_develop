#pragma once

#include <cstdarg>
#include <cstdint>
#include <string>

namespace alog {

class FileAppender;

constexpr uint32_t LOG_LEVEL_DISABLE = 0;
constexpr uint32_t LOG_LEVEL_FATAL = 1;
constexpr uint32_t LOG_LEVEL_ERROR = 2;
constexpr uint32_t LOG_LEVEL_WARN = 3;
constexpr uint32_t LOG_LEVEL_INFO = 4;
constexpr uint32_t LOG_LEVEL_DEBUG = 5;
constexpr uint32_t LOG_LEVEL_TRACE = 6;

class Logger {
public:
    static Logger *getLogger(const char * /*name*/) {
        static Logger inst;
        return &inst;
    }
    void setLevel(uint32_t /*level*/) {}
    void setAppender(FileAppender * /*appender*/) {}
    void setInheritFlag(bool /*flag*/) {}
    void log(uint32_t /*level*/, const char * /*fmt*/, ...) {}
    void logBinaryMessage(uint32_t /*level*/, const std::string & /*msg*/) {}
    uint32_t getLevel() { return LOG_LEVEL_INFO; }
    void flush() {}
    static void shutdown() {}
};

} // namespace alog
