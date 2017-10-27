//
//  mcts.hpp
//  kidney
//
//  Created by Vitor Baisi Hadad on 10/24/17.
//  Copyright © 2017 halflearned. All rights reserved.
//

#ifndef mcts_hpp
#define mcts_hpp

#include <memory>
#include <cmath>
#include <algorithm>
#include "environment.hpp"

using vertex_id = unsigned long;
using cycle = std::vector<vertex_id>;

cycle null_token{4294967295}; // Ugh.

struct MCTSNode {
    
    // "Nothing" constructor
    MCTSNode(vertex_id id = 0,  int t = 0, std::vector<cycle> acts = {{}})
    : parent(0), t(t), expandable(acts), id(id) {};
    
    // "Advance" and "Stay" constructor
    MCTSNode(vertex_id(id), vertex_id par, int t,  std::vector<cycle> exp, cycle tak, double r)
    : parent(par), t(t), id(id), expandable(exp), taken(tak), reward(r) {};
    
    // Copy constructor
    MCTSNode(const MCTSNode& v) = default;
    
    // Assignment
    MCTSNode& operator= (const MCTSNode& t) { return *this; };
    
    const vertex_id parent;
    const int t;
    int visits = 1;
    double reward = 0;
    const vertex_id id;
    cycle taken;
    std::vector<cycle> expandable;
    std::vector<vertex_id> children;
};


using MCTSTree = std::vector<MCTSNode>;

class MCTS {
public:
    MCTS(EnvGraph g, int t = 0, double sc = 0.7071, int max_depth = 100, int num_sims = 100, int num_rolls = 5) :
    environment(g), scalar(sc),
    max_tree_depth(t + max_depth), max_rollout_depth(t + max_depth),
    num_simulations(num_sims), num_rollouts(num_rolls) {
        auto acts = get_actions(t);
        tree.push_back(MCTSNode(0, t, acts));
    };
    EnvGraph environment;
    const double scalar;
    const int max_tree_depth;
    const int max_rollout_depth;
    const int num_simulations;
    const int num_rollouts;
    MCTSTree tree;
    void backup(vertex_id id, double reward);
    vertex_id expand(vertex_id v);
    vertex_id tree_policy();
    void simulate();
    
    vertex_id best_child(const vertex_id v) const;
    MCTSNode stay(const MCTSNode& v, const cycle action) const;
    MCTSNode advance(const MCTSNode& v, const cycle action) const;
    MCTSNode& root() { return tree[0]; }
    std::vector<cycle> get_actions(const int t) const;
    std::vector<cycle> get_actions(const int t, const EnvGraph env) const;
    double rollout(const vertex_id v) const;
    bool is_fully_expanded(const vertex_id v) const;
    bool is_root(const MCTSNode& v) const { return v.id == 0; };
    bool is_null_action(const cycle cyc) const { return cyc == null_token; };
    cycle pop_action(std::vector<cycle>& actions) const;
    cycle choose_action() const;
    double compute_score(const MCTSNode& node) const;
    static Solution run(EnvGraph env, double sc = 0.7071, int max_depth = 100, int num_sims = 100, int num_rolls = 5);
};


bool has_overlap(const cycle cyc, const cycle action) {
    for (auto&v : action) {
        if (std::find(cyc.begin(), cyc.end(), v) != cyc.end())
            return true;
    }
    return false;
}




cycle MCTS::pop_action(std::vector<cycle>& actions) const {
    assert(actions.size() > 0);
    auto taken_action = actions.back();
    std::vector<cycle> remaining_actions{};
    for (auto& cyc: actions) {
        if (not has_overlap(cyc, taken_action)) {
            remaining_actions.push_back(cyc);
        }
    }
    actions = remaining_actions;
    return taken_action;
}



double MCTS::rollout(const vertex_id v) const {
    EnvGraph env(environment);
    auto node = tree[v];
    env.populate_from(node.t, max_rollout_depth);
    int T = node.t + max_rollout_depth;
    //Solution greedy = solve_greedily(env, T);
    double reward = 0;
    for (int s = node.t; s < T; ++s) {
        auto actions = get_actions(s, env);
        while (actions.size() > 0) {
            auto rnd_cyc = pop_action(actions);
            if (!is_null_action(rnd_cyc)) {
                for (auto& v: rnd_cyc) {
                    env.match(v);
                    reward += 1.0;
                }
            } else {
                break;
            }
        }
    }
    double payoff = reward / max_rollout_depth; //(reward > greedy.obj) ? 1 : -1;
    //std::cout << "Payoff: " << reward << "\n"; // - " << greedy.obj << " = " << payoff << "\n";
    return payoff;
};



bool MCTS::is_fully_expanded(const vertex_id v) const {
    return tree[v].expandable.empty();
};


double MCTS::compute_score(const MCTSNode& node) const {
    auto total_visits = tree[node.parent].visits;
    auto exploit = node.reward / node.visits;
    auto explore = std::sqrt(2*std::log(total_visits)/node.visits);
    return exploit + scalar * explore;
};


vertex_id MCTS::best_child(const vertex_id v) const {
    assert(v < tree.size());
    assert(tree[v].children.size() > 0);
    
    std::default_random_engine rng{std::random_device{}()};
    auto children = tree[v].children;
    std::shuffle(children.begin(), children.end(), rng);
    
    return *std::max_element(children.begin(), children.end(),
                               [&](vertex_id id1, vertex_id id2) {
                                   return compute_score(tree[id1]) < compute_score(tree[id2]); });
};


std::vector<cycle> MCTS::get_actions(const int t) const {
    return get_actions(t, this->environment);
}


std::vector<cycle> MCTS::get_actions(const int t, const EnvGraph env) const {
    auto cs = env.get_two_cycles(t);
    cs.push_back(null_token);
    std::default_random_engine rng{std::random_device{}()};
    std::shuffle(cs.begin(), cs.end(), rng);
    assert(!cs.empty());
    return cs;
};




vertex_id MCTS::expand(vertex_id node_id) {
    cycle action = pop_action(tree[node_id].expandable);
    MCTSNode child = is_null_action(action)     ? advance(tree[node_id], action) :
                     is_fully_expanded(node_id) ? advance(tree[node_id], action) :
                     stay(tree[node_id], action);
    tree[node_id].children.push_back(child.id);
    tree.push_back(child);
    assert(!tree[node_id].children.empty());
    return child.id;
};



MCTSNode MCTS::advance(const MCTSNode& node, const cycle taken) const {
    auto acts = get_actions(node.t + 1);
    double r = taken == null_token ? 0 : taken.size();
    
    auto child = MCTSNode(tree.size(),     // Own id
                          node.id,         // Parent
                          node.t + 1,      // Time
                          acts,            // Expandable
                          taken,            // Taken action
                          r);               // Reward
    
    assert(child.id != child.parent);
    assert(child.parent == node.id);
    assert(!child.expandable.empty());
    return child;
};

MCTSNode MCTS::stay(const MCTSNode& node, const cycle taken) const {
    double r = taken.size();
    auto child = MCTSNode(tree.size(),     // Own id
                          node.id,         // Parent id
                          node.t,           // time
                          node.expandable,  // Expandable (modified inside expand function)
                          taken,            // Taken action
                          r);               // Reward
    
    assert(child.id != child.parent);
    assert(child.parent == node.id);
    assert(!child.expandable.empty());
    return child;
};




vertex_id MCTS::tree_policy() {
    vertex_id node_id = 0; // Start at root
    while (tree[node_id].t < max_tree_depth) {
        if (!is_fully_expanded(node_id)) {
            auto child = expand(node_id);
            assert(!tree[node_id].children.empty());
            return child;
        } else {
            assert(is_fully_expanded(node_id));
            assert(tree[node_id].children.size() > 0);
            node_id = best_child(node_id);
        }
    }
    return node_id;
};



void MCTS::backup(vertex_id id, double reward) {
    while (true) {
        tree[id].visits++;
        tree[id].reward += reward;
        if (is_root(tree[id])) return;
        id = tree[tree[id].parent].id;
    }
};


void MCTS::simulate() {
    for (int i=0; i < num_simulations; i++) {
        vertex_id node_id = tree_policy();
        double r = 0;
        for (int k=0; k < num_rollouts; k++) r += rollout(node_id);
        r /= num_rollouts;
        //std::cout << "Avg payoff: " << r <<  "\n";
        backup(node_id, r);
    }
}


cycle MCTS::choose_action() const {
    assert(tree[0].children.size() > 0);
    auto most_visited_id = std::max_element(tree[0].children.begin(), tree[0].children.end(),
                                         [&](vertex_id id1, vertex_id id2) {
                                             return tree[id1].visits < tree[id2].visits;
                                         });
    return tree[*most_visited_id].taken;
};



Solution MCTS::run(EnvGraph env,
                   double sc,
                   int max_depth,
                   int num_sims,
                   int num_rolls) {
    
    std::set<vertex_id> matched;
    double obj = 0;
    double reward = 0;
    int t = 0;
    int period = -1;
    while (t < env.T) {
        if (period != t) std::cout << "\n\n\nPERIOD" << ++period;
        auto mcts = MCTS(env, t, sc, max_depth, num_sims, num_rolls);
        if (mcts.root().expandable[0] == null_token and mcts.root().expandable.size() == 1) {
            std::cout << "Only null action left at time " << t << "\n";
            t++; continue;
        }
        mcts.simulate();
        assert(!mcts.tree[0].children.empty());
        auto taken = mcts.choose_action();
        std::cout << "Children scores: \n";
        for (auto&c : mcts.tree[0].children) {
            std::cout << "( " << c << ", "
            << mcts.tree[c].visits << ", "
            << mcts.tree[c].reward << ", "
            << mcts.compute_score(mcts.tree[c]) << ", "
            << (mcts.tree[c].taken == null_token) << ")\n ";
        }
        std::cout << "Took null action: " << ((taken == null_token) ? "YES" : "No") << "\n";
        if (taken != null_token) {
            for (auto&v : taken) {
                matched.insert(v);
                env.match(v);
                reward += 1;
                obj += 1;
            }
        } else {
            if (reward == 0) {
                
            }
            //std::cout << "\n\nPERIOD: " << t << ", Reward: " << reward << "\n";
            t++;
            reward = 0;
        }
    }
    return Solution(obj, matched);
}





#endif /* mcts_hpp */
