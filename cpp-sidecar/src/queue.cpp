#include "queue.hpp"

void RequestQueue::push(Request req) {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        queue_.push(std::move(req));
    }
    // Wake one waiting consumer.
    cv_.notify_one();
}

std::optional<Request> RequestQueue::pop() {
    std::unique_lock<std::mutex> lock(mtx_);

    // Wait until there is work, or we've been told to stop.
    cv_.wait(lock, [this] {
        return !queue_.empty() || stopped_;
    });

    if (queue_.empty()) {
        // Shutdown was signalled and the queue is drained.
        return std::nullopt;
    }

    // std::priority_queue only offers const top() — we must cast away const
    // to move out of it, then pop.
    Request req = std::move(const_cast<Request&>(queue_.top()));
    queue_.pop();
    return req;
}

void RequestQueue::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        stopped_ = true;
    }
    // Wake ALL blocked consumers so they can exit.
    cv_.notify_all();
}

std::size_t RequestQueue::depth() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return queue_.size();
}
