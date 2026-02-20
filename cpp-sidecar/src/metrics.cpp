#include "metrics.hpp"

#include <sstream>
#include <iomanip>

// ── LatencyHistogram ────────────────────────────────────────────────────────

void LatencyHistogram::record(double ms) {
    std::lock_guard<std::mutex> lock(mtx_);
    sum_   += ms;
    total_ += 1;

    // Increment every bucket whose upper bound is >= ms (cumulative histogram).
    for (std::size_t i = 0; i < BUCKETS.size(); ++i) {
        if (ms <= BUCKETS[i]) {
            counts_[i]++;
        }
    }
    // The "+Inf" bucket always increments.
    counts_[BUCKETS.size()]++;
}

std::string LatencyHistogram::to_prometheus(const std::string& name,
                                             const std::string& help) const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::ostringstream oss;

    oss << "# HELP " << name << " " << help << "\n";
    oss << "# TYPE " << name << " histogram\n";

    for (std::size_t i = 0; i < BUCKETS.size(); ++i) {
        oss << name << "_bucket{le=\"" << std::fixed << std::setprecision(0)
            << BUCKETS[i] << "\"} " << counts_[i] << "\n";
    }
    oss << name << "_bucket{le=\"+Inf\"} " << counts_[BUCKETS.size()] << "\n";
    oss << name << "_sum "   << std::fixed << std::setprecision(3) << sum_   << "\n";
    oss << name << "_count " << total_ << "\n";

    return oss.str();
}

// ── Metrics ─────────────────────────────────────────────────────────────────

std::string Metrics::to_prometheus() const {
    std::ostringstream oss;

    // http_requests_total
    oss << "# HELP http_requests_total Total number of HTTP requests received.\n";
    oss << "# TYPE http_requests_total counter\n";
    oss << "http_requests_total " << requests_total.load() << "\n\n";

    // queue_depth
    oss << "# HELP queue_depth Current number of requests waiting in the queue.\n";
    oss << "# TYPE queue_depth gauge\n";
    oss << "queue_depth " << queue_depth.load() << "\n\n";

    // inference_latency_ms
    oss << inference_latency.to_prometheus(
            "inference_latency_ms",
            "End-to-end inference latency in milliseconds (queue wait + UDS round-trip).")
        << "\n";

    // queue_wait_ms
    oss << queue_wait.to_prometheus(
            "queue_wait_ms",
            "Time in milliseconds a request spent waiting in the queue.")
        << "\n";

    return oss.str();
}
