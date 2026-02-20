#include "server.hpp"
#include "queue.hpp"
#include "metrics.hpp"

#include <boost/asio/strand.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>

#include <nlohmann/json.hpp>

#include <chrono>
#include <iostream>
#include <memory>
#include <string>

namespace beast = boost::beast;
namespace http  = beast::http;
namespace net   = boost::asio;
using tcp       = net::ip::tcp;
using json      = nlohmann::json;

// ── Per-connection session ───────────────────────────────────────────────────

// Each accepted TCP connection gets a Session that owns the socket and
// drives the full request/response lifecycle asynchronously.
class Session : public std::enable_shared_from_this<Session> {
public:
    Session(tcp::socket                   socket,
            std::shared_ptr<RequestQueue> queue,
            std::shared_ptr<Metrics>      metrics)
        : stream_(std::move(socket))
        , queue_(std::move(queue))
        , metrics_(std::move(metrics))
    {}

    void start() { read_request(); }

private:
    // ── Step 1: async read ───────────────────────────────────────────────────
    void read_request() {
        auto self = shared_from_this();
        http::async_read(
            stream_, buffer_, req_,
            [self](beast::error_code ec, std::size_t) {
                if (ec) return;  // client disconnected or parse error
                self->handle_request();
            });
    }

    // ── Step 2: routing ──────────────────────────────────────────────────────
    void handle_request() {
        // Count every request regardless of route.
        metrics_->requests_total.fetch_add(1, std::memory_order_relaxed);

        const auto& target = req_.target();

        if (req_.method() == http::verb::get && target == "/health") {
            return send_response(http::status::ok, "text/plain", "OK\n");
        }

        if (req_.method() == http::verb::get && target == "/metrics") {
            return send_response(http::status::ok,
                                 "text/plain; version=0.0.4; charset=utf-8",
                                 metrics_->to_prometheus());
        }

        if (req_.method() == http::verb::post && target == "/infer") {
            return handle_infer();
        }

        send_response(http::status::not_found, "text/plain",
                      "404 Not Found. Valid endpoints: POST /infer, GET /metrics, GET /health\n");
    }

    // ── Step 3: inference dispatch ───────────────────────────────────────────
    void handle_infer() {
        // Parse request body as JSON.
        json body;
        try {
            body = json::parse(req_.body());
        } catch (const json::exception& e) {
            return send_response(http::status::bad_request, "application/json",
                                 R"({"error":"Invalid JSON: )" + std::string(e.what()) + "\"}");
        }

        if (!body.contains("prompt") || !body["prompt"].is_string()) {
            return send_response(http::status::bad_request, "application/json",
                                 R"({"error":"Missing or non-string 'prompt' field"})");
        }

        Request req;
        req.id         = body.value("id",         generate_id());
        req.prompt     = body["prompt"].get<std::string>();
        req.max_tokens = body.value("max_tokens", 128);
        req.priority   = body.value("priority",   0);

        // Capture the future *before* moving the request into the queue,
        // because after the move req.response_promise is no longer accessible.
        auto future = req.response_promise.get_future();

        // Track queue depth.
        metrics_->queue_depth.fetch_add(1, std::memory_order_relaxed);
        queue_->push(std::move(req));

        // Wait for the worker thread to fulfill the promise.
        // This blocks the current Asio strand — acceptable because the
        // io_context runs on multiple threads; other connections are unaffected.
        // For a fully non-blocking design, this could be replaced with
        // std::async + beast::post back onto the strand.
        std::string response_json;
        try {
            response_json = future.get();
        } catch (const std::exception& e) {
            metrics_->queue_depth.fetch_sub(1, std::memory_order_relaxed);
            return send_response(http::status::internal_server_error,
                                 "application/json",
                                 std::string(R"({"error":")") + e.what() + "\"}");
        }

        metrics_->queue_depth.fetch_sub(1, std::memory_order_relaxed);
        send_response(http::status::ok, "application/json", response_json);
    }

    // ── Step 4: async write ──────────────────────────────────────────────────
    void send_response(http::status       status,
                       std::string_view   content_type,
                       std::string        body) {
        auto res = std::make_shared<http::response<http::string_body>>();
        res->version(req_.version());
        res->keep_alive(false);
        res->result(status);
        res->set(http::field::server, "inference-proxy/1.0");
        res->set(http::field::content_type, content_type);
        res->body() = std::move(body);
        res->prepare_payload();

        auto self = shared_from_this();
        http::async_write(
            stream_, *res,
            [self, res](beast::error_code ec, std::size_t) {
                self->stream_.socket().shutdown(tcp::socket::shutdown_send, ec);
            });
    }

    // ── Helpers ──────────────────────────────────────────────────────────────
    static std::string generate_id() {
        // Simple incrementing ID. For production use a proper UUID library.
        static std::atomic<uint64_t> counter{0};
        return "req-" + std::to_string(counter.fetch_add(1));
    }

    beast::tcp_stream                  stream_;
    beast::flat_buffer                 buffer_;
    http::request<http::string_body>   req_;
    std::shared_ptr<RequestQueue>      queue_;
    std::shared_ptr<Metrics>           metrics_;
};

// ── HttpServer ───────────────────────────────────────────────────────────────

HttpServer::HttpServer(net::io_context&              ioc,
                       unsigned short                 port,
                       std::shared_ptr<RequestQueue>  queue,
                       std::shared_ptr<Metrics>       metrics)
    : ioc_(ioc)
    , acceptor_(ioc, tcp::endpoint(tcp::v4(), port))
    , queue_(std::move(queue))
    , metrics_(std::move(metrics))
{}

void HttpServer::start() {
    acceptor_.set_option(net::socket_base::reuse_address(true));
    accept();
}

void HttpServer::accept() {
    acceptor_.async_accept(
        net::make_strand(ioc_),
        [this](beast::error_code ec, tcp::socket socket) {
            if (!ec) {
                std::make_shared<Session>(
                    std::move(socket), queue_, metrics_)->start();
            }
            // Keep accepting regardless of individual connection errors.
            accept();
        });
}
