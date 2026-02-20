#include "queue.hpp"
#include "metrics.hpp"
#include "server.hpp"
#include "uds_client.hpp"

#include <boost/asio/io_context.hpp>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace net = boost::asio;
using json    = nlohmann::json;

// ── Configuration (read from environment variables) ──────────────────────────

static std::string env_str(const char* name, std::string default_val) {
    if (const char* val = std::getenv(name)) return val;
    return default_val;
}

static int env_int(const char* name, int default_val) {
    if (const char* val = std::getenv(name)) return std::atoi(val);
    return default_val;
}

// ── Worker thread function ───────────────────────────────────────────────────
//
// Each worker thread:
//   1. Pops the highest-priority Request from the shared queue.
//   2. Serialises it to JSON, sends to the Python worker via UDS.
//   3. Parses the JSON response, fulfils the promise so the HTTP session
//      can return the result to the client.
//   4. Updates latency metrics.
//
// Exits cleanly when queue.pop() returns std::nullopt (shutdown).

static void worker_loop(std::shared_ptr<RequestQueue> queue,
                        std::shared_ptr<Metrics>      metrics,
                        const std::string&            socket_path) {
    UDSClient uds(socket_path);

    while (true) {
        // Block until work arrives or shutdown is signalled.
        auto maybe_req = queue->pop();
        if (!maybe_req) break;  // shutdown

        Request& req = *maybe_req;

        // Build the JSON payload to send to the Python worker.
        json payload;
        payload["id"]         = req.id;
        payload["prompt"]     = req.prompt;
        payload["max_tokens"] = req.max_tokens;

        std::string response_json;
        bool        had_error = false;

        auto start = std::chrono::steady_clock::now();
        try {
            response_json = uds.send_request(payload.dump());
        } catch (const std::exception& e) {
            had_error = true;
            // Build an error JSON that the HTTP layer will forward to the client.
            json err;
            err["id"]    = req.id;
            err["error"] = e.what();
            err["text"]  = "";
            response_json = err.dump();
            std::cerr << "[worker] UDS error for request " << req.id
                      << ": " << e.what() << "\n";
        }
        auto end = std::chrono::steady_clock::now();

        // Record inference latency regardless of error status.
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        metrics->inference_latency.record(ms);

        if (!had_error) {
            // Sanity-check that the Python worker echoed back the same ID.
            try {
                auto resp = json::parse(response_json);
                if (resp.value("id", "") != req.id) {
                    std::cerr << "[worker] WARNING: ID mismatch. Expected "
                              << req.id << " got " << resp.value("id", "?") << "\n";
                }
            } catch (...) { /* non-fatal */ }
        }

        // Fulfil the promise — wakes the waiting HTTP session.
        req.response_promise.set_value(std::move(response_json));
    }

    std::cout << "[worker] thread exiting cleanly.\n";
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    // Configuration
    const std::string socket_path  = env_str("SOCKET_PATH",  "/tmp/inference.sock");
    const int         http_port    = env_int("HTTP_PORT",     8080);
    const int         num_workers  = env_int("NUM_WORKERS",   4);
    const int         io_threads   = env_int("IO_THREADS",    2);

    std::cout << "[main] Starting inference proxy\n"
              << "       HTTP port   : " << http_port    << "\n"
              << "       Socket path : " << socket_path  << "\n"
              << "       Worker threads: " << num_workers << "\n"
              << "       I/O threads : " << io_threads   << "\n";

    auto queue   = std::make_shared<RequestQueue>();
    auto metrics = std::make_shared<Metrics>();

    // ── Start worker threads ─────────────────────────────────────────────────
    // std::jthread automatically joins on destruction (C++20).
    std::vector<std::jthread> workers;
    workers.reserve(num_workers);
    for (int i = 0; i < num_workers; ++i) {
        workers.emplace_back(worker_loop, queue, metrics, socket_path);
    }

    // ── Start HTTP server ────────────────────────────────────────────────────
    net::io_context ioc{io_threads};
    HttpServer server(ioc, static_cast<unsigned short>(http_port), queue, metrics);
    server.start();

    std::cout << "[main] HTTP server listening on port " << http_port << "\n";

    // Run the io_context on multiple threads for concurrent connection handling.
    std::vector<std::thread> io_pool;
    io_pool.reserve(io_threads - 1);
    for (int i = 0; i < io_threads - 1; ++i) {
        io_pool.emplace_back([&ioc] { ioc.run(); });
    }
    ioc.run();  // current thread also runs the io_context

    // Graceful shutdown: signal workers and wait for jthreads to join.
    queue->shutdown();
    // workers vector goes out of scope here → jthreads join automatically.

    for (auto& t : io_pool) t.join();

    std::cout << "[main] Shutdown complete.\n";
    return 0;
}
