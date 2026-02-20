#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

// Lightweight histogram with fixed buckets (in milliseconds).
// All operations are thread-safe via a single mutex.
class LatencyHistogram {
public:
    // Upper bounds (inclusive) of each bucket, in milliseconds.
    static constexpr std::array<double, 8> BUCKETS{
        50, 100, 250, 500, 1000, 2000, 5000, 10000
    };

    void record(double ms);

    // Returns Prometheus-formatted histogram lines for the given metric name.
    std::string to_prometheus(const std::string& name,
                              const std::string& help) const;

private:
    mutable std::mutex           mtx_;
    std::array<uint64_t, 9>     counts_{};   // 8 buckets + "+Inf"
    double                       sum_{0.0};
    uint64_t                     total_{0};
};

// Central metrics store shared between the HTTP server and worker threads.
// Atomic counters for simple values; histogram for latency.
class Metrics {
public:
    // Incremented by the HTTP acceptor on every incoming request.
    std::atomic<uint64_t> requests_total{0};

    // Maintained by RequestQueue::push/pop.
    std::atomic<uint64_t> queue_depth{0};

    // Records end-to-end inference latency (queue wait + UDS round-trip).
    LatencyHistogram inference_latency;

    // Records time a request spent waiting in the queue before being picked up.
    LatencyHistogram queue_wait;

    // Format all metrics as a Prometheus text exposition string.
    std::string to_prometheus() const;
};

// RAII timer: records elapsed milliseconds into a LatencyHistogram on destruction.
class ScopedTimer {
public:
    explicit ScopedTimer(LatencyHistogram& hist)
        : hist_(hist), start_(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        hist_.record(ms);
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    LatencyHistogram&                         hist_;
    std::chrono::steady_clock::time_point     start_;
};
