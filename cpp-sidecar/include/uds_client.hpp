#pragma once

#include <cstdint>
#include <string>
#include <stdexcept>

// Exception thrown when UDS communication fails.
class UDSError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Synchronous Unix Domain Socket client.
//
// Opens a new connection per request — simple and correct for our use case
// where the Python worker is single-threaded and handles one request at a time.
// Each send_request() call:
//   1. Connects to socket_path_.
//   2. Writes a 4-byte little-endian length prefix + JSON payload.
//   3. Reads a 4-byte length prefix + JSON response.
//   4. Closes the connection.
//
// Thread-safety: individual instances are safe to call from one thread.
// Use one UDSClient per worker thread.
class UDSClient {
public:
    explicit UDSClient(std::string socket_path);

    // Send a JSON request string, block until the response JSON is received.
    // Throws UDSError on any socket or I/O failure.
    std::string send_request(const std::string& json_payload);

private:
    // Write exactly n bytes, retrying on EINTR.
    static void write_all(int fd, const char* buf, std::size_t n);

    // Read exactly n bytes, retrying on EINTR.
    static void read_all(int fd, char* buf, std::size_t n);

    std::string socket_path_;
};
