#pragma once

#include <condition_variable>
#include <future>
#include <mutex>
#include <optional>
#include <queue>
#include <stop_token>
#include <string>

// A single inference request traveling through the system.
// The response_promise is fulfilled by the worker thread after UDS round-trip.
struct Request {
    int         priority{0};    // higher value = served first
    std::string id;             // UUID assigned at HTTP ingress
    std::string prompt;
    int         max_tokens{128};

    // Non-copyable because std::promise is non-copyable.
    std::promise<std::string> response_promise;

    // Move-only
    Request() = default;
    Request(Request&&) = default;
    Request& operator=(Request&&) = default;
    Request(const Request&) = delete;
    Request& operator=(const Request&) = delete;
};

// Comparator: higher priority → served first (max-heap).
struct RequestComparator {
    bool operator()(const Request& a, const Request& b) const {
        return a.priority < b.priority;  // inverted for max-heap
    }
};

// Thread-safe priority queue implementing the Producer-Consumer pattern.
//
// Producers (HTTP handler threads) call push().
// Consumers (worker threads)     call pop(), which blocks until an item
//                                is available or shutdown() is called.
class RequestQueue {
public:
    RequestQueue() = default;

    // Push a request onto the queue (called from HTTP handler threads).
    // Thread-safe.
    void push(Request req);

    // Block until a request is available, then return it.
    // Returns std::nullopt when shutdown() has been called and the queue
    // is drained — signals the consumer to exit its loop.
    std::optional<Request> pop();

    // Signal all blocked pop() callers to wake up and return nullopt.
    void shutdown();

    // Current number of items waiting (approximate; for metrics only).
    std::size_t depth() const;

private:
    mutable std::mutex                                           mtx_;
    std::condition_variable                                      cv_;
    std::priority_queue<Request, std::vector<Request>,
                        RequestComparator>                       queue_;
    bool                                                         stopped_{false};
};
