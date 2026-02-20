#pragma once

#include "metrics.hpp"
#include "queue.hpp"

#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <memory>
#include <string>

// Forward declaration to keep the header light.
namespace boost::asio { class io_context; }

// Boost.Beast async HTTP server.
//
// Accepts TCP connections on the given port and dispatches requests:
//   POST /infer   → validates JSON, pushes Request onto queue, awaits response
//   GET  /metrics → returns Prometheus text exposition
//   GET  /health  → returns 200 OK (used by k8s liveness probe)
//
// Each connection is handled on the io_context strand — no per-connection
// threads. The io_context should be run() on multiple threads externally for
// full concurrency.
class HttpServer {
public:
    HttpServer(boost::asio::io_context&        ioc,
               unsigned short                   port,
               std::shared_ptr<RequestQueue>    queue,
               std::shared_ptr<Metrics>         metrics);

    // Begin accepting connections (non-blocking, posts work onto ioc).
    void start();

private:
    void accept();

    boost::asio::io_context&           ioc_;
    boost::asio::ip::tcp::acceptor     acceptor_;
    std::shared_ptr<RequestQueue>      queue_;
    std::shared_ptr<Metrics>           metrics_;
};
