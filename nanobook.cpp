// nanobook.cpp
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

// ============================
// Minimal SPSC Ring Buffer
// - power-of-2 capacity
// - leaves 1 slot empty to distinguish full vs empty
// ============================
template <typename T>
class SpscRing {
public:
    explicit SpscRing(size_t capacity_pow2)
        : mask_(capacity_pow2 - 1), buf_(capacity_pow2) {
        // capacity must be power of 2
        if ((capacity_pow2 & mask_) != 0) throw runtime_error("capacity not power of 2");
        head_.store(0, memory_order_relaxed);
        tail_.store(0, memory_order_relaxed);
    }

    bool try_push(const T& v) {
        const size_t head = head_.load(memory_order_relaxed);
        const size_t next = head + 1;
        const size_t tail = tail_.load(memory_order_acquire);

        // full if next would make (head - tail) reach capacity
        if ((next - tail) >= buf_.size()) return false;

        buf_[head & mask_] = v;
        head_.store(next, memory_order_release);
        return true;
    }

    bool try_pop(T& out) {
        const size_t tail = tail_.load(memory_order_relaxed);
        if (tail == head_.load(memory_order_acquire)) return false; // empty
        out = buf_[tail & mask_];
        tail_.store(tail + 1, memory_order_release);
        return true;
    }

    size_t size() const {
        const size_t h = head_.load(memory_order_acquire);
        const size_t t = tail_.load(memory_order_acquire);
        return h - t;
    }

    size_t capacity() const { return buf_.size(); }

private:
    const size_t mask_;
    vector<T> buf_;
    atomic<size_t> head_{0}, tail_{0};
};

// ============================
// Types & Messages
// ============================
using ns = chrono::nanoseconds;
using steady = chrono::steady_clock;

enum class Side : uint8_t { Buy = 0, Sell = 1 };
enum class MsgType : uint8_t { New = 0, Cancel = 1, Market = 2 };

struct Msg {
    MsgType type;
    Side side;
    uint32_t order_id;   // unique
    double price;        // ignored for Market
    uint32_t qty;        // shares
    uint32_t symbol;     // single symbol demo: 0
    uint64_t t_ns;       // enqueue timestamp (ns since start)
};

struct Fill {
    uint32_t maker_id;
    uint32_t taker_id;
    double price;
    uint32_t qty;
};

// ============================
// Order Book (price-time priority)
// ============================
struct Order {
    uint32_t id;
    Side side;
    double price;
    uint32_t qty;
    // No heap new/delete in hot path (but containers may allocate).
};

struct PriceLevel {
    deque<Order> fifo;
    uint32_t total = 0;
};

struct Book {
    // best bid = begin() of bids, best ask = begin() of asks
    map<double, PriceLevel, std::less<double>> asks;            // low->high
    map<double, PriceLevel, std::greater<double>> bids;         // high->low
    // id -> (price, side) to enable cancel in O(logN) + O(level scan)
    unordered_map<uint32_t, pair<double, Side>> id_index;
};

// ============================
// Matching Engine
// ============================
class MatchingEngine {
public:
    explicit MatchingEngine(bool emit_fills = false) : emit_fills_(emit_fills) {}

    // returns matched qty (for metrics)
    uint32_t on_limit(Order o, vector<Fill>& out_fills) {
        uint32_t filled = 0;
        if (o.side == Side::Buy) {
            while (o.qty && !book_.asks.empty()) {
                auto it = book_.asks.begin();
                if (it->first > o.price) break;
                filled += cross(o, it->second, it->first, out_fills);
                if (it->second.total == 0) book_.asks.erase(it);
            }
            if (o.qty) add_to_book(o);
        } else {
            while (o.qty && !book_.bids.empty()) {
                auto it = book_.bids.begin();
                if (it->first < o.price) break;
                filled += cross(o, it->second, it->first, out_fills);
                if (it->second.total == 0) book_.bids.erase(it);
            }
            if (o.qty) add_to_book(o);
        }
        return filled;
    }

    uint32_t on_market(Order o, vector<Fill>& out_fills) {
        uint32_t filled = 0;
        if (o.side == Side::Buy) {
            while (o.qty && !book_.asks.empty()) {
                auto it = book_.asks.begin();
                filled += cross(o, it->second, it->first, out_fills);
                if (it->second.total == 0) book_.asks.erase(it);
            }
        } else {
            while (o.qty && !book_.bids.empty()) {
                auto it = book_.bids.begin();
                filled += cross(o, it->second, it->first, out_fills);
                if (it->second.total == 0) book_.bids.erase(it);
            }
        }
        return filled;
    }

    bool on_cancel(uint32_t id) {
        auto it = book_.id_index.find(id);
        if (it == book_.id_index.end()) return false;
        auto [px, side] = it->second;
        if (side == Side::Buy) {
            auto pit = book_.bids.find(px);
            if (pit == book_.bids.end()) return false;
            if (!erase_by_id(pit->second, id)) return false;
            if (pit->second.total == 0) book_.bids.erase(pit);
        } else {
            auto pit = book_.asks.find(px);
            if (pit == book_.asks.end()) return false;
            if (!erase_by_id(pit->second, id)) return false;
            if (pit->second.total == 0) book_.asks.erase(pit);
        }
        book_.id_index.erase(it);
        return true;
    }

    optional<pair<double, uint32_t>> best_bid() const {
        if (book_.bids.empty()) return nullopt;
        auto it = book_.bids.begin();
        return {{it->first, it->second.total}};
    }
    optional<pair<double, uint32_t>> best_ask() const {
        if (book_.asks.empty()) return nullopt;
        auto it = book_.asks.begin();
        return {{it->first, it->second.total}};
    }

private:
    Book book_;
    bool emit_fills_;

    void add_to_book(const Order& o) {
        if (o.side == Side::Buy) {
            auto& level = book_.bids[o.price];
            level.fifo.push_back(o);
            level.total += o.qty;
            book_.id_index[o.id] = {o.price, o.side};
        } else {
            auto& level = book_.asks[o.price];
            level.fifo.push_back(o);
            level.total += o.qty;
            book_.id_index[o.id] = {o.price, o.side};
        }
    }

    uint32_t cross(Order& taker, PriceLevel& level, double trade_px, vector<Fill>& out_fills) {
        uint32_t matched = 0;
        while (taker.qty && !level.fifo.empty()) {
            Order& maker = level.fifo.front();
            uint32_t q = min(taker.qty, maker.qty);
            taker.qty -= q;
            maker.qty -= q;
            matched += q;
            level.total -= q;
            if (emit_fills_) out_fills.push_back({maker.id, taker.id, trade_px, q});
            if (maker.qty == 0) {
                book_.id_index.erase(maker.id);
                level.fifo.pop_front();
            }
        }
        return matched;
    }

    static bool erase_by_id(PriceLevel& level, uint32_t id) {
        for (auto it = level.fifo.begin(); it != level.fifo.end(); ++it) {
            if (it->id == id) {
                level.total -= it->qty;
                level.fifo.erase(it);
                return true;
            }
        }
        return false;
    }
};

// ============================
// Benchmark / Modes
// ============================
enum class RunMode : uint8_t { Throughput = 0, Latency = 1 };

static RunMode parse_mode(const string& s) {
    if (s == "throughput") return RunMode::Throughput;
    if (s == "latency") return RunMode::Latency;
    throw runtime_error("invalid --mode (use throughput|latency)");
}

// ============================
// Feed Producer (synthetic)
// ============================
struct BenchCfg {
    uint64_t n_msgs = 1'000'000;

    // sampling / measurement
    uint64_t warmup_msgs = 50'000;
    uint64_t sample_every = 100;

    // mode
    RunMode mode = RunMode::Throughput;
    size_t target_q_depth = 1024; // only used in latency mode (best-effort throttling)

    // csv
    bool csv = false;
    string csv_path = "perf.csv";
};

class Producer {
public:
    Producer(SpscRing<Msg>& q, const steady::time_point& t0, const BenchCfg& cfg)
        : q_(q), t0_(t0), cfg_(cfg) {}

    void operator()() {
        // Roughly 60% New LIMITs, 20% MARKET, 20% Cancel
        mt19937_64 rng(42);
        uniform_int_distribution<int> dist100(0, 99);
        uniform_int_distribution<int> qty(1, 1000);
        uniform_real_distribution<double> px(99.0, 101.0);
        uint32_t next_id = 1;

        for (uint64_t i = 0; i < cfg_.n_msgs; i++) {
            // Latency mode: attempt to avoid huge queue backlog
            if (cfg_.mode == RunMode::Latency) {
                while (q_.size() > cfg_.target_q_depth) {
                    std::this_thread::yield();
                }
            }

            Msg m{};
            int r = dist100(rng);
            if (r < 60) {
                m.type = MsgType::New;
                m.side = (dist100(rng) < 50) ? Side::Buy : Side::Sell;
                m.order_id = next_id++;
                m.price = floor(px(rng) * 100.0) / 100.0;
                m.qty = qty(rng);
            } else if (r < 80) {
                m.type = MsgType::Market;
                m.side = (dist100(rng) < 50) ? Side::Buy : Side::Sell;
                m.order_id = next_id++;
                m.price = 0.0;
                m.qty = qty(rng);
            } else {
                // Cancel an earlier id (best effort)
                m.type = MsgType::Cancel;
                m.side = Side::Buy;
                m.order_id = (next_id > 1) ? (1 + (rng() % (next_id - 1))) : 1;
                m.price = 0.0;
                m.qty = 0;
            }

            m.symbol = 0;
            m.t_ns = (uint64_t)chrono::duration_cast<ns>(steady::now() - t0_).count();

            while (!q_.try_push(m)) std::this_thread::yield();
        }
    }

private:
    SpscRing<Msg>& q_;
    steady::time_point t0_;
    const BenchCfg& cfg_;
};

// ============================
// Consumer / Dispatcher
// ============================
static uint64_t pct_sorted(const vector<uint64_t>& v_sorted, double p) {
    if (v_sorted.empty()) return 0;
    size_t k = size_t(p * double(v_sorted.size() - 1));
    return v_sorted[k];
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    BenchCfg cfg;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "--bench" && i + 1 < argc) cfg.n_msgs = strtoull(argv[++i], nullptr, 10);
        else if (a == "--csv" && i + 1 < argc) { cfg.csv = true; cfg.csv_path = argv[++i]; }
        else if (a == "--mode" && i + 1 < argc) cfg.mode = parse_mode(argv[++i]);
        else if (a == "--warmup" && i + 1 < argc) cfg.warmup_msgs = strtoull(argv[++i], nullptr, 10);
        else if (a == "--sample-every" && i + 1 < argc) cfg.sample_every = strtoull(argv[++i], nullptr, 10);
        else if (a == "--target-q" && i + 1 < argc) cfg.target_q_depth = (size_t)strtoull(argv[++i], nullptr, 10);
        else if (a == "--help") {
            cout <<
                "Usage:\n"
                "  ./nanobook [--bench N] [--mode throughput|latency] [--warmup N] [--sample-every N] [--target-q N]\n"
                "            [--csv out.csv]\n\n"
                "Examples:\n"
                "  ./nanobook --bench 1000000\n"
                "  ./nanobook --bench 5000000 --mode latency --target-q 1024\n"
                "  ./nanobook --bench 5000000 --csv perf.csv\n";
            return 0;
        }
    }

    // ~1M slots. Effective max occupancy is capacity-1 due to full/empty rule.
    SpscRing<Msg> q(1 << 20);
    MatchingEngine eng(false);

    const auto t0 = steady::now();
    thread prod(Producer(q, t0, cfg));

    Msg m;
    vector<Fill> fills;
    fills.reserve(16);

    // Latency samples
    vector<uint64_t> e2e_lat_ns;     // enqueue -> end of processing (includes queueing)
    vector<uint64_t> eng_lat_ns;     // dequeue -> end of processing (engine-only)

    const uint64_t expected_samples =
        (cfg.n_msgs > cfg.warmup_msgs && cfg.sample_every)
            ? ((cfg.n_msgs - cfg.warmup_msgs) / cfg.sample_every + 16)
            : 16;
    e2e_lat_ns.reserve((size_t)min<uint64_t>(expected_samples, 300000));
    eng_lat_ns.reserve((size_t)min<uint64_t>(expected_samples, 300000));

    // Queue depth stats (approx)
    size_t max_q_depth = 0;

    uint64_t processed = 0;

    // Benchmark window: start measuring throughput after warmup so it aligns with latency window
    steady::time_point measure_start = t0;
    bool measurement_started = false;

    while (processed < cfg.n_msgs) {
        if (q.try_pop(m)) {
            const auto t_deq = steady::now();

            // Start measurement window after warmup completes
            processed++;
            if (!measurement_started && processed >= cfg.warmup_msgs) {
                measurement_started = true;
                measure_start = t_deq;
            }

            // track approximate backlog (after pop)
            max_q_depth = max(max_q_depth, q.size());

            // dispatch
            fills.clear();
            if (m.type == MsgType::New) {
                Order o{m.order_id, m.side, m.price, m.qty};
                eng.on_limit(o, fills);
            } else if (m.type == MsgType::Market) {
                Order o{m.order_id, m.side, 0.0, m.qty};
                eng.on_market(o, fills);
            } else {
                eng.on_cancel(m.order_id);
            }

            const auto t_done = steady::now();

            // Sampling after warmup
            if (measurement_started && cfg.sample_every && (processed % cfg.sample_every) == 0) {
                const uint64_t done_ns = (uint64_t)chrono::duration_cast<ns>(t_done - t0).count();
                const uint64_t deq_ns  = (uint64_t)chrono::duration_cast<ns>(t_deq  - t0).count();

                // end-to-end (enqueue -> done)
                e2e_lat_ns.push_back(done_ns - m.t_ns);
                // engine-only (dequeue -> done)
                eng_lat_ns.push_back(done_ns - deq_ns);
            }
        } else {
            std::this_thread::yield();
        }
    }

    prod.join();
    const auto t_end = steady::now();

    // Throughput: measure over the post-warmup window
    double secs = 0.0;
    uint64_t measured_msgs = 0;
    if (cfg.n_msgs > cfg.warmup_msgs && measurement_started) {
        measured_msgs = cfg.n_msgs - cfg.warmup_msgs;
        secs = chrono::duration<double>(t_end - measure_start).count();
    } else {
        measured_msgs = cfg.n_msgs;
        secs = chrono::duration<double>(t_end - t0).count();
    }
    const double mps = (secs > 0.0) ? (double(measured_msgs) / secs) : 0.0;

    sort(e2e_lat_ns.begin(), e2e_lat_ns.end());
    sort(eng_lat_ns.begin(), eng_lat_ns.end());

    const uint64_t e2e_p50 = pct_sorted(e2e_lat_ns, 0.50);
    const uint64_t e2e_p90 = pct_sorted(e2e_lat_ns, 0.90);
    const uint64_t e2e_p99 = pct_sorted(e2e_lat_ns, 0.99);

    const uint64_t eng_p50 = pct_sorted(eng_lat_ns, 0.50);
    const uint64_t eng_p90 = pct_sorted(eng_lat_ns, 0.90);
    const uint64_t eng_p99 = pct_sorted(eng_lat_ns, 0.99);

    auto bb = eng.best_bid();
    auto ba = eng.best_ask();

    cout << "Processed: " << cfg.n_msgs << " msgs"
         << " (warmup " << cfg.warmup_msgs << ")"
         << " in " << fixed << setprecision(3) << secs << "s (measured window)\n";
    cout << "Mode: " << ((cfg.mode == RunMode::Throughput) ? "throughput" : "latency")
         << "  target_q=" << cfg.target_q_depth << "\n";
    cout << "Throughput: " << setprecision(0) << mps << " msgs/s\n";
    cout << "MaxQueueDepth (approx): " << max_q_depth << " (capacity " << (q.capacity() - 1) << ")\n";
    cout << "E2E Latency (ns):   p50=" << e2e_p50 << "  p90=" << e2e_p90 << "  p99=" << e2e_p99 << "\n";
    cout << "Engine Latency (ns): p50=" << eng_p50 << "  p90=" << eng_p90 << "  p99=" << eng_p99 << "\n";
    if (bb) cout << "BestBid: " << bb->first << " x " << bb->second << "\n";
    if (ba) cout << "BestAsk: " << ba->first << " x " << ba->second << "\n";

    if (cfg.csv) {
        ofstream f(cfg.csv_path);
        f << "mode,warmup_msgs,sample_every,target_q_depth,max_queue_depth,";
        f << "e2e_p50_ns,e2e_p90_ns,e2e_p99_ns,";
        f << "eng_p50_ns,eng_p90_ns,eng_p99_ns,";
        f << "throughput_msgs_per_s,measured_msgs,total_msgs\n";
        f << ((cfg.mode == RunMode::Throughput) ? "throughput" : "latency") << ",";
        f << cfg.warmup_msgs << ",";
        f << cfg.sample_every << ",";
        f << cfg.target_q_depth << ",";
        f << max_q_depth << ",";
        f << e2e_p50 << "," << e2e_p90 << "," << e2e_p99 << ",";
        f << eng_p50 << "," << eng_p90 << "," << eng_p99 << ",";
        f << (uint64_t)mps << "," << measured_msgs << "," << cfg.n_msgs << "\n";
        f.close();
        cout << "Wrote CSV to " << cfg.csv_path << "\n";
    }

    return 0;
}