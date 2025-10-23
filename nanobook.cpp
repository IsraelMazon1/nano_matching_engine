#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>
#include <string>
using namespace std;

// ============================
// Minimal SPSC Ring Buffer
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
        if (next - tail_.load(memory_order_acquire) > buf_.size()) return false; // full
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

enum class Side : uint8_t { Buy=0, Sell=1 };
enum class MsgType : uint8_t { New=0, Cancel=1, Market=2 };

struct Msg {
    MsgType type;
    Side side;
    uint32_t order_id;    // unique
    double price;         // ignored for Market
    uint32_t qty;         // shares
    uint32_t symbol;      // single symbol demo: 0
    uint64_t t_ns;        // enqueue timestamp (ns since start)
};

struct Fill {
    uint32_t maker_id;
    uint32_t taker_id;
    double   price;
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
    // No heap new/delete in hot path.
};

struct PriceLevel {
    deque<Order> fifo;
    uint32_t total = 0;
};

struct Book {
    // best bid = rbegin() of bids, best ask = begin() of asks
    map<double, PriceLevel, std::less<double>> asks; // low->high
    map<double, PriceLevel, std::greater<double>> bids; // high->low
    // id -> iterator to enable cancel in O(logN)
    unordered_map<uint32_t, pair<double, Side>> id_index;
};

// ============================
// Matching Engine
// ============================
class MatchingEngine {
public:
    explicit MatchingEngine(bool emit_fills=false) : emit_fills_(emit_fills) {}

    // returns matched qty (for metrics)
    uint32_t on_limit(Order o, vector<Fill>& out_fills) {
        uint32_t filled = 0;
        if (o.side == Side::Buy) {
            // match against best ask while price <= o.price
            while (o.qty && !book_.asks.empty()) {
                auto it = book_.asks.begin();
                if (it->first > o.price) break;
                filled += cross(o, it->second, it->first, out_fills);
                if (it->second.total == 0) book_.asks.erase(it);
            }
            if (o.qty) add_to_book(o);
        } else {
            // Sell vs best bid while price >= o.price
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
        // price ignored; sweep until qty consumed
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

    optional<pair<double,uint32_t>> best_bid() const {
        if (book_.bids.empty()) return nullopt;
        auto it = book_.bids.begin();
        return {{it->first, it->second.total}};
    }
    optional<pair<double,uint32_t>> best_ask() const {
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
                // remove maker from index and queue
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
// Feed Producer (synthetic)
// ============================
struct BenchCfg {
    uint64_t n_msgs = 1'000'000;
    bool csv = false;
    string csv_path = "perf.csv";
};

// payload is Msg; producer stamps relative ns since start
class Producer {
public:
    Producer(SpscRing<Msg>& q, const steady::time_point& t0, BenchCfg cfg)
        : q_(q), t0_(t0), cfg_(cfg) {}

    void operator()() {
        // Generate roughly 60% New LIMITs, 20% MARKET, 20% Cancel
        mt19937_64 rng(42);
        uniform_int_distribution<int> dist100(0,99);
        uniform_int_distribution<int> qty(1, 1000);
        uniform_real_distribution<double> px(99.0, 101.0);
        uint32_t next_id = 1;

        for (uint64_t i=0;i<cfg_.n_msgs;i++) {
            Msg m{};
            int r = dist100(rng);
            if (r < 60) {
                m.type = MsgType::New;
                m.side = (dist100(rng) < 50) ? Side::Buy : Side::Sell;
                m.order_id = next_id++;
                m.price = floor(px(rng)*100.0)/100.0;
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
};

// ============================
// Consumer / Dispatcher
// ============================
struct Perf {
    vector<uint64_t> samples_ns; // latency samples (subset)
    uint64_t processed = 0;
};

static uint64_t pct(const vector<uint64_t>& v, double p) {
    if (v.empty()) return 0;
    size_t k = size_t(p * (v.size()-1));
    return v[k];
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    BenchCfg cfg;
    for (int i=1;i<argc;i++) {
        string a = argv[i];
        if (a == "--bench" && i+1<argc) cfg.n_msgs = strtoull(argv[++i], nullptr, 10);
        else if (a == "--csv" && i+1<argc) { cfg.csv = true; cfg.csv_path = argv[++i]; }
    }

    SpscRing<Msg> q(1<<20); // ~1M slots
    MatchingEngine eng(false);
    Perf perf;

    const auto t0 = steady::now();
    thread prod(Producer(q, t0, cfg));

    Msg m;
    vector<Fill> fills; fills.reserve(16);
    // Sample every Nth message for latency to reduce overhead
    const uint64_t sample_every = 100;
    vector<uint64_t> lat;
    lat.reserve(min<uint64_t>(cfg.n_msgs/ sample_every + 10, 200000));

    uint64_t last_log = 0;
    while (perf.processed < cfg.n_msgs) {
        if (q.try_pop(m)) {
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
    if (bb) cout << "BestBid: " << bb->first << " x " << bb->second << "\n";
    if (ba) cout << "BestAsk: " << ba->first << " x " << ba->second << "\n";

    if (cfg.csv) {
        ofstream f(cfg.csv_path);
        f << "p50_ns,p90_ns,p99_ns,throughput_msgs_per_s,total_msgs\n";
        f << p50 << "," << p90 << "," << p99 << "," << (uint64_t)mps << "," << perf.processed << "\n";
        f.close();
        cout << "Wrote CSV to " << cfg.csv_path << "\n";
    }
    return 0;
}
