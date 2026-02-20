#include "uds_client.hpp"

#include <cerrno>
#include <cstring>
#include <stdexcept>

// POSIX socket headers (available in Linux containers)
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

UDSClient::UDSClient(std::string socket_path)
    : socket_path_(std::move(socket_path)) {}

std::string UDSClient::send_request(const std::string& json_payload) {
    // 1. Create socket file descriptor.
    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        throw UDSError(std::string("socket() failed: ") + std::strerror(errno));
    }

    // RAII guard — close fd on any early return.
    struct FdGuard {
        int fd;
        ~FdGuard() { if (fd >= 0) ::close(fd); }
    } guard{fd};

    // 2. Connect to the Python worker's socket.
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    // socket_path_ must fit in sun_path (108 bytes on Linux).
    if (socket_path_.size() >= sizeof(addr.sun_path)) {
        throw UDSError("Socket path too long: " + socket_path_);
    }
    std::strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        throw UDSError(std::string("connect() to '") + socket_path_ +
                       "' failed: " + std::strerror(errno));
    }

    // 3. Send: 4-byte little-endian length prefix + JSON payload.
    auto payload_len = static_cast<uint32_t>(json_payload.size());
    // Convert to little-endian manually for portability.
    uint8_t len_buf[4] = {
        static_cast<uint8_t>( payload_len        & 0xFF),
        static_cast<uint8_t>((payload_len >>  8) & 0xFF),
        static_cast<uint8_t>((payload_len >> 16) & 0xFF),
        static_cast<uint8_t>((payload_len >> 24) & 0xFF),
    };
    write_all(fd, reinterpret_cast<char*>(len_buf), 4);
    write_all(fd, json_payload.data(), json_payload.size());

    // 4. Receive: 4-byte length prefix then JSON response.
    uint8_t resp_len_buf[4];
    read_all(fd, reinterpret_cast<char*>(resp_len_buf), 4);

    uint32_t resp_len =
        static_cast<uint32_t>(resp_len_buf[0])        |
        (static_cast<uint32_t>(resp_len_buf[1]) <<  8) |
        (static_cast<uint32_t>(resp_len_buf[2]) << 16) |
        (static_cast<uint32_t>(resp_len_buf[3]) << 24);

    if (resp_len == 0 || resp_len > 10 * 1024 * 1024) {
        throw UDSError("Invalid response length: " + std::to_string(resp_len));
    }

    std::string response(resp_len, '\0');
    read_all(fd, response.data(), resp_len);

    return response;
}

void UDSClient::write_all(int fd, const char* buf, std::size_t n) {
    std::size_t written = 0;
    while (written < n) {
        ssize_t ret = ::write(fd, buf + written, n - written);
        if (ret < 0) {
            if (errno == EINTR) continue;  // interrupted by signal, retry
            throw UDSError(std::string("write() failed: ") + std::strerror(errno));
        }
        written += static_cast<std::size_t>(ret);
    }
}

void UDSClient::read_all(int fd, char* buf, std::size_t n) {
    std::size_t bytes_read = 0;
    while (bytes_read < n) {
        ssize_t ret = ::read(fd, buf + bytes_read, n - bytes_read);
        if (ret < 0) {
            if (errno == EINTR) continue;
            throw UDSError(std::string("read() failed: ") + std::strerror(errno));
        }
        if (ret == 0) {
            throw UDSError("Connection closed by Python worker before full response received.");
        }
        bytes_read += static_cast<std::size_t>(ret);
    }
}
