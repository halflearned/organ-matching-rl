//
// environment.hpp
//  kidney
//
//  Created by Vitor Baisi Hadad on 10/17/17.
//  Copyright © 2017 halflearned. All rights reserved.
//

#ifndef environment_h
#define environment_h

//#include <pybind11/pybind11.h>
//#include <Eigen/Dense>
#include <random>
#include <vector>
#include <tuple>
#include <boost/iterator/filter_iterator.hpp>
#include <cassert>
#include "solver.hpp"




using vertex_id = unsigned long;

struct EnvNode {
    EnvNode(int entry, int death, int patient, int donor, double pra, unsigned long id) :
    entry(entry), death(death), patient(patient), donor(donor), pra(pra), id(id) {};
    int patient;
    int donor;
    int entry;
    int death;
    double pra;
    vertex_id id;
    bool matched = false;
    std::vector<vertex_id> out_edges;
};

using EnvNodes = std::vector<EnvNode>;
using cycle = std::vector<unsigned long>;


class EnvGraph  {
public:
    EnvNodes nodes;
    EnvGraph(double e, double d, int T):
    entry_rate(e), death_rate(d), T(T) {
        dist_n_entries = std::poisson_distribution<int>(e);
        dist_sojourn = std::geometric_distribution<int>(d);
        dist_pra_type = std::discrete_distribution<int>{.7019, 0.2, 0.0981};
        dist_blood_type = std::discrete_distribution<int>{.4814, 0.3373, .1428, 0.0385};
        populate();
        draw_edges();
    };
    int T;
    double entry_rate;
    double death_rate;
    void populate();
    void populate_from(int beg, int horiz);
    void draw_edges();
    void match(vertex_id v);
    bool can_donate(EnvNode& from, EnvNode& to);
    std::vector<cycle> get_two_cycles() const;
    std::vector<cycle> get_two_cycles(int t) const;
private:
    std::poisson_distribution<int> dist_n_entries;
    std::geometric_distribution<int> dist_sojourn;
    std::discrete_distribution<int> dist_blood_type {.4814, 0.3373, .1428, 0.0385};
    std::vector<double> pra_values{.05, .45, .95};
    std::discrete_distribution<int> dist_pra_type {.7019, 0.2, 0.0981};
    std::uniform_real_distribution<double> dist_unif{0.0, 1.0};
    
    std::default_random_engine rng;
    int get_n_entries() { return dist_n_entries(rng); };
    int get_sojourn() { return dist_sojourn(rng) + 1; };
    int get_blood_type() { return dist_blood_type(rng); };
    double get_pra() { return pra_values[dist_pra_type(rng)]; };
    double get_unif() { return dist_unif(rng); };
};

void EnvGraph::match(vertex_id v) {
    assert(v < nodes.size());
    assert(!nodes[v].matched);
    nodes[v].matched = true;
}



void EnvGraph::populate() {
    populate_from(0, T);
}



void EnvGraph::populate_from(int t, int horiz) {
    int max_time = t + horiz;
    unsigned long id;
    if (nodes.size() > 0) {
        auto beg = std::find_if(nodes.begin(), nodes.end(),
                                [t](const EnvNode& v){return v.entry == t; });
        nodes.erase(beg, nodes.end());
        id = (beg != nodes.end()) ? beg->id : nodes.back().id;
    } else {
        id = 0;
    }
    for (int s = t; s < max_time; ++s) {
        auto n_today = get_n_entries();
        for (int i = 0; i < n_today; ++i) {
            auto n = EnvNode(s,                  // Entry
                             s + get_sojourn(),  // Death
                             get_blood_type(),   // Patient
                             get_blood_type(),   // Donor
                             get_pra(),          // PRA
                             id++);              // ID
            nodes.push_back(n);
        }
    }
}


bool EnvGraph::can_donate(EnvNode& from, EnvNode& to) {
    bool not_same = from.id != to.id;
    if (!not_same) return false;
    bool contemporaneous = (from.entry <= to.death) and (from.death >= to.entry);
    if (!contemporaneous) return false;
    bool hist_compat = from.pra < get_unif();
    if (!hist_compat) return false;
    bool blood_compat = (from.donor == to.patient) or
                        (to.patient == 3) or
                        (from.donor == 0);
    if (!blood_compat) return false;
    return true;
};


    
void EnvGraph::draw_edges() {
    for (auto&u: nodes) {
        for (auto&v: nodes) {
            if (can_donate(u, v)) {
                u.out_edges.push_back(v.id);
            }
        }
    }
}


std::vector<cycle> EnvGraph::get_two_cycles() const{
    std::vector<cycle> two_cycles;
    auto pred = [](const EnvNode& a, const EnvNode& b){
        return (!a.matched) and (!b.matched) and
                (a.id < b.id) and
                (std::find(a.out_edges.begin(), a.out_edges.end(), b.id) != a.out_edges.end()) and
                (std::find(b.out_edges.begin(), b.out_edges.end(), a.id) != b.out_edges.end());
    };
    
    for (auto& u: nodes) {
        for (auto &v: nodes) {
            if (pred(u, v)) {
                two_cycles.push_back(std::vector<unsigned long>{u.id, v.id});
            }
        }
    }
    return two_cycles;
}



std::vector<cycle> EnvGraph::get_two_cycles(int t) const {
    std::vector<cycle> two_cycles;
    auto pred = [t](const EnvNode& a, const EnvNode& b){
        return  (!a.matched) and (!b.matched) and
                (a.id < b.id) and
                (a.entry <= t and a.death >= t) and
                (b.entry <= t and b.death >= t) and
                (std::find(a.out_edges.begin(), a.out_edges.end(), b.id) != a.out_edges.end()) and
                (std::find(b.out_edges.begin(), b.out_edges.end(), a.id) != b.out_edges.end()); };
    
    for (auto& u: nodes) {
        for (auto &v: nodes) {
            if (pred(u, v)) {
                two_cycles.push_back(std::vector<unsigned long>{u.id, v.id});
            }
        }
    }
    return two_cycles;
}


        
Solution solve_optimally(const EnvGraph& g) {
    auto cs = g.get_two_cycles();
    auto n = g.nodes.size();
    auto solver = Solver(cs, n);
    return solver.solve();
}


Solution solve_greedily(EnvGraph g) { // Must copy graph
    double obj = 0;
    std::set<unsigned long> chosen;
    for (int t = 0; t < g.T; ++t) {
        auto cs = g.get_two_cycles(t);
        auto n = g.nodes.size();
        auto solver = Solver(cs, n);
        auto solution = solver.solve();
        for (auto &v: solution.chosen) {
            assert(!g.nodes[v].matched);
            g.match(v);
        }
        obj += solution.obj;
    }
    return Solution(obj, chosen);
}






#endif /* environment_h */

