#pragma once

#include <charconv>
#include <string>
#include <string_view>
#include <system_error>

namespace autil {
namespace StringUtil {

template <typename T>
bool fromString(const char *str, T &ret) {
    if (!str) {
        return false;
    }
    std::string_view sv(str);
    auto r = std::from_chars(sv.data(), sv.data() + sv.size(), ret);
    return r.ec == std::errc() && r.ptr == sv.data() + sv.size();
}

inline bool fromString(const char *str, bool &ret) {
    if (!str) {
        return false;
    }
    std::string s(str);
    if (s == "1" || s == "true" || s == "True" || s == "TRUE") {
        ret = true;
        return true;
    }
    if (s == "0" || s == "false" || s == "False" || s == "FALSE") {
        ret = false;
        return true;
    }
    return false;
}

} // namespace StringUtil
} // namespace autil
