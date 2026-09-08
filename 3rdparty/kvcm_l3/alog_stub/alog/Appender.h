#pragma once

#include <string>

namespace alog {

class PatternLayout {
public:
    void setLogPattern(const std::string & /*pattern*/) {}
};

class FileAppender {
public:
    static FileAppender *getAppender(const char * /*name*/) {
        static FileAppender inst;
        return &inst;
    }
    void setMaxSize(int /*size*/) {}
    void setCacheLimit(int /*limit*/) {}
    void setHistoryLogKeepCount(int /*count*/) {}
    void setAsyncFlush(bool /*flag*/) {}
    void setCompress(bool /*flag*/) {}
    void setLayout(PatternLayout * /*layout*/) {}
};

} // namespace alog
