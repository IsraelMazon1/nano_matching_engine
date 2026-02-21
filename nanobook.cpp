<<<<<<< HEAD
// nanobook.cpp
=======
>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
<<<<<<< HEAD
#include <cstdint>
=======
>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <random>
<<<<<<< HEAD
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

=======
#include <thread>
#include <unordered_map>
#include <vector>
#include <string>
>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24
using namespace std;

// ============================
// Minimal SPSC Ring Buffer
<<<<<<< HEAD
// - power-of-2 capacity
// - leaves 1 slot empty to distinguish full vs empty
=======
>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24
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
<<<<<<< HEAD

    bool try_push(const T& v) {
        const size_t head = head_.load(memory_order_relaxed);
        const size_t next = head + 1;
        const size_t tail = tail_.load(memory_order_acquire);

        // full if next would make (head - tail) reach capacity
        if ((next - tail) >= buf_.size()) return false;

=======
    bool try_push(const T& v) {
        const size_t head = head_.load(memory_order_relaxed);
        const size_t next = head + 1;
        if (next - tail_.load(memory_order_acquire) > buf_.size()) return false; // full
>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24
        buf_[head & mask_] = v;
        head_.store(next, memory_order_release);
        return true;
    }
<<<<<<< HEAD

=======
>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24
    bool try_pop(T& out) {
        const size_t tail = tail_.load(memory_order_relaxed);
        if (tail == head_.load(memory_order_acquire)) return false; // empty
        out = buf_[tail & mask_];
        tail_.store(tail + 1, memory_order_release);
        return true;
    }
<<<<<<< HEAD

=======
>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24
    size_t size() const {
        const size_t h = head_.load(memory_order_acquire);
        const size_t t = tail_.load(memory_order_acquire);
        return h - t;
    }
<<<<<<< HEAD

    size_t capacity() const { return buf_.size(); }

=======
>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24
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

<<<<<<< HEAD
enum class Side : uint8_t { Buy = 0, Sell = 1 };
enum class MsgType : uint8_t { New = 0, Cancel = 1, Market = 2 };
=======
enum class Side : uint8_t { Buy=0, Sell=1 };
enum class MsgType : uint8_t { New=0, Cancel=1, Market=2 };
>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24

struct Msg {
    MsgType type;
    Side side;
<<<<<<< HEAD
    uint32_t order_id;   // unique
    double price;        // ignored for Market
    uint32_t qty;        // shares
    uint32_t symbol;     // single symbol demo: 0
    uint64_t t_ns;       // enqueue timestamp (ns since start)
=======
    uint32_t order_id;    // unique
    double price;         // ignored for Market
    uint32_t qty;         // shares
    uint32_t symbol;      // single symbol demo: 0
    uint64_t t_ns;        // enqueue timestamp (ns since start)
>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24
};

struct Fill {
    uint32_t maker_id;
    uint32_t taker_id;
<<<<<<< HEAD
    double price;
=======
    double   price;
>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24
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
<<<<<<< HEAD
    // No heap new/delete in hot path (but containers may allocate).
=======
    // No heap new/delete in hot path.
>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24
};

struct PriceLevel {
    deque<Order> fifo;
    uint32_t total = 0;
};

struct Book {
<<<<<<< HEAD
    // best bid = begin() of bids, best ask = begin() of asks
    map<double, PriceLevel, std::less<double>> asks;            // low->high
    map<double, PriceLevel, std::greater<double>> bids;         // high->low
    // id -> (price, side) to enable cancel in O(logN) + O(level scan)
=======
    // best bid = rbegin() of bids, best ask = begin() of asks
    map<double, PriceLevel, std::less<double>> asks; // low->high
    map<double, PriceLevel, std::greater<double>> bids; // high->low
    // id -> iterator to enable cancel in O(logN)
>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24
    unordered_map<uint32_t, pair<double, Side>> id_index;
};

// ============================
// Matching Engine
// ============================
class MatchingEngine {
public:
<<<<<<< HEAD
    explicit MatchingEngine(bool emit_fills = false) : emit_fills_(emit_fills) {}
=======
    explicit MatchingEngine(bool emit_fills=false) : emit_fills_(emit_fills) {}
>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24

    // returns matched qty (for metrics)
    uint32_t on_limit(Order o, vector<Fill>& out_fills) {
        uint32_t filled = 0;
        if (o.side == Side::Buy) {
<<<<<<< HEAD
=======
            // match against best ask while price <= o.price
>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24
            while (o.qty && !book_.asks.empty()) {
                auto it = book_.asks.begin();
                if (it->first > o.price) break;
                filled += cross(o, it->second, it->first, out_fills);
                if (it->second.total == 0) book_.asks.erase(it);
            }
            if (o.qty) add_to_book(o);
        } else {
<<<<<<< HEAD
=======
            // Sell vs best bid while price >= o.price
>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24
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
<<<<<<< HEAD
=======
        // price ignored; sweep until qty consumed
>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24
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

<<<<<<< HEAD
    optional<pair<double, uint32_t>> best_bid() const {
=======
    optional<pair<double,uint32_t>> best_bid() const {
>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24
        if (book_.bids.empty()) return nullopt;
        auto it = book_.bids.begin();
        return {{it->first, it->second.total}};
    }
<<<<<<< HEAD
    optional<pair<double, uint32_t>> best_ask() const {
=======
    optional<pair<double,uint32_t>> best_ask() const {
>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24
        if (book_.asks.empty()) return nullopt;
        auto it = book_.asks.begin();
        return {{it->first, it->second.total}};
    }

private:
    Book book_;
    bool emit_fills_;

    void add_to_book(const Order& o) {
<<<<<<< HEAD
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
=======
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

>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24

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
<<<<<<< HEAD
=======
                // remove maker from index and queue
>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24
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
<<<<<<< HEAD
// Benchmark / Modes
// ============================
enum class RunMode : uint8_t { Throughput = 0, Latency = 1 };

static RunMode parse_mode(const string& s) {
    if (s == "throughput") return RunMode::Throughput;
    if (s == "latency") return RunMode::Latency;
    throw runtime_error("invalid --mode (use throughput|latency)");
}

// ============================
=======
>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24
// Feed Producer (synthetic)
// ============================
struct BenchCfg {
    uint64_t n_msgs = 1'000'000;
<<<<<<< HEAD

    // sampling / measurement
    uint64_t warmup_msgs = 50'000;
    uint64_t sample_every = 100;

    // mode
    RunMode mode = RunMode::Throughput;
    size_t target_q_depth = 1024; // only used in latency mode (best-effort throttling)

    // csv
=======
>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24
    bool csv = false;
    string csv_path = "perf.csv";
};

<<<<<<< HEAD
class Producer {
public:
    Producer(SpscRing<Msg>& q, const steady::time_point& t0, const BenchCfg& cfg)
        : q_(q), t0_(t0), cfg_(cfg) {}

    void operator()() {
        // Roughly 60% New LIMITs, 20% MARKET, 20% Cancel
        mt19937_64 rng(42);
        uniform_int_distribution<int> dist100(0, 99);
=======
// payload is Msg; producer stamps relative ns since start
class Producer {
public:
    Producer(SpscRing<Msg>& q, const steady::time_point& t0, BenchCfg cfg)
        : q_(q), t0_(t0), cfg_(cfg) {}

    void operator()() {
        // Generate roughly 60% New LIMITs, 20% MARKET, 20% Cancel
        mt19937_64 rng(42);
        uniform_int_distribution<int> dist100(0,99);
>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24
        uniform_int_distribution<int> qty(1, 1000);
        uniform_real_distribution<double> px(99.0, 101.0);
        uint32_t next_id = 1;

<<<<<<< HEAD
        for (uint64_t i = 0; i < cfg_.n_msgs; i++) {
            // Latency mode: attempt to avoid huge queue backlog
            if (cfg_.mode == RunMode::Latency) {
                while (q_.size() > cfg_.target_q_depth) {
                    std::this_thread::yield();
                }
            }

=======
        for (uint64_t i=0;i<cfg_.n_msgs;i++) {
>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24
            Msg m{};
            int r = dist100(rng);
            if (r < 60) {
                m.type = MsgType::New;
                m.side = (dist100(rng) < 50) ? Side::Buy : Side::Sell;
                m.order_id = next_id++;
<<<<<<< HEAD
                m.price = floor(px(rng) * 100.0) / 100.0;
=======
                m.price = floor(px(rng)*100.0)/100.0;
>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24
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
<<<<<<< HEAD
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
=======
                m.order_id = (next_id>1) ? (1 + (rng() % (next_id-1))) : 1;
                m.price = 0.0;
                m.qty = 0;
            }
            m.symbol = 0;
            m.t_ns = (uint64_t)chrono::duration_cast<ns>(steady::now() - t0_).count();

            // backoff if full
            while (!q_.try_push(m)) std::this_thread::yield();
        }
    }
private:
    SpscRing<Msg>& q_;
    steady::time_point t0_;
    BenchCfg cfg_;
>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24
};

// ============================
// Consumer / Dispatcher
// ============================
<<<<<<< HEAD
static uint64_t pct_sorted(const vector<uint64_t>& v_sorted, double p) {
    if (v_sorted.empty()) return 0;
    size_t k = size_t(p * double(v_sorted.size() - 1));
    return v_sorted[k];
=======
struct Perf {
    vector<uint64_t> samples_ns; // latency samples (subset)
    uint64_t processed = 0;
};

static uint64_t pct(const vector<uint64_t>& v, double p) {
    if (v.empty()) return 0;
    size_t k = size_t(p * (v.size()-1));
    return v[k];
>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    BenchCfg cfg;
<<<<<<< HEAD
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
=======
    for (int i=1;i<argc;i++) {
        string a = argv[i];
        if (a == "--bench" && i+1<argc) cfg.n_msgs = strtoull(argv[++i], nullptr, 10);
        else if (a == "--csv" && i+1<argc) { cfg.csv = true; cfg.csv_path = argv[++i]; }
    }

    SpscRing<Msg> q(1<<20); // ~1M slots
    MatchingEngine eng(false);
    Perf perf;
>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24

    const auto t0 = steady::now();
    thread prod(Producer(q, t0, cfg));

    Msg m;
<<<<<<< HEAD
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

=======
    vector<Fill> fills; fills.reserve(16);
    // Sample every Nth message for latency to reduce overhead
    const uint64_t sample_every = 100;
    vector<uint64_t> lat;
    lat.reserve(min<uint64_t>(cfg.n_msgs/ sample_every + 10, 200000));

    uint64_t last_log = 0;
    while (perf.processed < cfg.n_msgs) {
        if (q.try_pop(m)) {
>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24
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

<<<<<<< HEAD
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
=======
            perf.processed++;
            if ((perf.processed % sample_every)==0) {
                auto tnow = steady::now();
                uint64_t now_ns = (uint64_t)chrono::duration_cast<ns>(tnow - t0).count();
                uint64_t one = now_ns - m.t_ns;
                lat.push_back(one);
            }

            // (optional) periodic heartbeat to avoid watchdogs
            if (perf.processed - last_log >= 1'000'000) {
                last_log = perf.processed;
            }
        } else {
            // idle wait
            std::this_thread::yield();
        }
    }
    prod.join();
    sort(lat.begin(), lat.end());
    auto t1 = steady::now();

    double secs = chrono::duration<double>(t1 - t0).count();
    double mps  = double(perf.processed)/secs;

    uint64_t p50 = pct(lat, 0.50);
    uint64_t p90 = pct(lat, 0.90);
    uint64_t p99 = pct(lat, 0.99);
    auto bb = eng.best_bid();
    auto ba = eng.best_ask();

    cout << "Processed: " << perf.processed << " msgs in " << fixed << setprecision(3) << secs << "s\n";
    cout << "Throughput: " << setprecision(0) << mps << " msgs/s\n";
    cout << "Latency (ns): p50=" << p50 << "  p90=" << p90 << "  p99=" << p99 << "\n";
>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24
    if (bb) cout << "BestBid: " << bb->first << " x " << bb->second << "\n";
    if (ba) cout << "BestAsk: " << ba->first << " x " << ba->second << "\n";

    if (cfg.csv) {
        ofstream f(cfg.csv_path);
<<<<<<< HEAD
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
=======
        f << "p50_ns,p90_ns,p99_ns,throughput_msgs_per_s,total_msgs\n";
        f << p50 << "," << p90 << "," << p99 << "," << (uint64_t)mps << "," << perf.processed << "\n";
        f.close();
        cout << "Wrote CSV to " << cfg.csv_path << "\n";
    }
    return 0;
}
>>>>>>> f562fd318667a2874a431fe4da1d255a226ddf24
