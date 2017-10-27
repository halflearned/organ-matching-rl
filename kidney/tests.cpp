//////
//////  environment_test.cpp
//////  kidney
//////
//////  Created by Vitor Baisi Hadad on 10/17/17.
//////  Copyright © 2017 halflearned. All rights reserved.
//////
////
////


#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "solver.hpp"
#include <iostream>
#include "environment.hpp"
#include "mcts.hpp"
//


TEST_CASE("Simple initialization", "EnvGraph") {
    EnvGraph g(17, .05, 23);
    REQUIRE(g.entry_rate == 17);
    REQUIRE(g.death_rate == .05);
    REQUIRE(g.T == 23);
    REQUIRE(g.nodes.size() > 0);
    for (auto& v: g.nodes) {
        REQUIRE(v.death > v.entry);
    }
}


TEST_CASE("Checking node attributes", "Env") {
    EnvGraph g(1, .5, 20);
    REQUIRE(g.nodes.size() > 0);
    int id = 0;
    for (auto& v: g.nodes) {
        REQUIRE(v.id == id++);
        REQUIRE(v.entry < 20);
        REQUIRE(v.entry < v.death);
        REQUIRE(v.patient >= 0);
        REQUIRE(v.patient <= 3);
        REQUIRE(v.donor >= 0);
        REQUIRE(v.donor <= 3);
        REQUIRE(((v.pra == 0.05) || (v.pra == 0.95) || (v.pra == 0.45)));
        REQUIRE(!v.matched);
    }
}




TEST_CASE("Can donate") {
    EnvGraph g(0, .1, 0);
    auto v = EnvNode(0, 100, 3, 0, 0, 0);
    auto u = EnvNode(0, 100, 3, 0, 0, 1);
    REQUIRE(g.can_donate(v, u));
    // ID is same - do not donate
    { auto v = EnvNode(0, 100, 3, 0, 0, 1);
    auto u = EnvNode(0, 100, 3, 0, 0, 1);
    REQUIRE(!g.can_donate(v, u)); }
    // Not contemporaneous - do not donate
    {auto v = EnvNode(50, 100, 3, 0, 0, 0);
    auto u = EnvNode(0, 49, 3, 0, 0, 1);
    REQUIRE(!g.can_donate(v, u)); }
    // AB (Blood type 3) patient always receives
    { for (int i = 0; i < 4; i++) {
        auto v = EnvNode(0, 100, 0, i, 0, 0);
        auto u = EnvNode(0, 100, 3, 0, 0, 1);
        REQUIRE(g.can_donate(v, u));
    } }
    // O (Blood type 0) patient always donates
    { for (int i = 0; i < 4; i++) {
        auto v = EnvNode(0, 100, 0, 0, 0, 0);
        auto u = EnvNode(0, 100, i, 0, 0, 1);
        REQUIRE(g.can_donate(v, u));
    } }
}


//TEST_CASE("Edges and cycles") {
//    EnvGraph g(5, .1, 10);
//    for (auto& v: g.nodes) {
//        std::cout << "Node id" << v.id;
//        std::cout << "  entry: " << v.entry;
//        std::cout << "  death: " << v.death;
//        std::cout << "  pat: " << v.patient;
//        std::cout << "  don: "   << v.donor;
//        std::cout << "  pra: "   << v.pra;
//        std::cout << "  out_edges: ";
//        for (auto &o: v.out_edges) {
//            std::cout << o << " ";
//        }
//        std::cout << "\n";
//    }
//}


TEST_CASE("Creating empty solver") {
    cycles cs{};
    auto s = Solver(cs, 2);
}


TEST_CASE("Solving a simple gurobi program") {
    cycles cs{{1,2}, {2,1}, {3,0}, {0,3}, {0,4}};
    auto solver = Solver(cs, 5);
    Solution sol = solver.solve();
    REQUIRE(sol.obj == 4);
    REQUIRE(sol.chosen.size() == 4);
}


TEST_CASE("Solving a small kidney exchange") {
    auto g = EnvGraph(15, .01, 10);
    cycles cs = g.get_two_cycles();
    REQUIRE(cs.size() > 0);
    auto n = g.nodes.size();
    auto solver = Solver(cs, n);
    Solution sol = solver.solve();
    REQUIRE(sol.obj > 0);
    REQUIRE(sol.chosen.size() <= n);
    REQUIRE(sol.chosen.size() == sol.obj);
}



TEST_CASE("Solving KEP optimally and greedily") {
    auto g = EnvGraph(10, .1, 10);
    auto opt = solve_optimally(g);
    auto greedy = solve_greedily(g);
    REQUIRE(greedy.obj <= opt.obj);
}


TEST_CASE("Creating an MCTSNode") {
    MCTSNode();
}

TEST_CASE("Creating an empty MCTS graph") {
    EnvGraph g(0, .1, 0);
    MCTS mcts(g);
    REQUIRE(mcts.tree.size() == 1);
}

TEST_CASE("Advance and stay") {

    EnvGraph g(0, .1, 0);
    MCTS mcts(g);
    auto& root = mcts.tree[0];
    auto child = mcts.advance(root, null_token);
    REQUIRE(child.t == 1);
    REQUIRE(child.visits == 1);
    REQUIRE(child.reward == 0);
    REQUIRE(child.parent == 0);

    cycle a{1,2};
    auto child2 = mcts.stay(root, a);
    REQUIRE(child2.t == 0);
    REQUIRE(child2.visits == 1);
    REQUIRE(child2.reward > 0);
    REQUIRE(child2.taken == a);
    REQUIRE(child2.parent == 0);
}

TEST_CASE("expansions", "MCTS") {
    EnvGraph g(5, .1, 10);
    MCTS mcts(g, 5);
    
    int expected_id = 1;
    int expected_treesize = 2;
    int expected_children_size = 1;
    while (mcts.root().expandable.size() > 0) {
        auto expandable_size = mcts.root().expandable.size();
        auto child_id = mcts.expand(0);
        auto child = mcts.tree[child_id];
        REQUIRE(mcts.root().children.size() == expected_children_size++);
        REQUIRE(mcts.root().expandable.size() < expandable_size);
        REQUIRE(child.id == expected_id++);
        REQUIRE(child.visits == 1);
        auto r = (child.taken == null_token) ? 0 : 2;
        REQUIRE(child.reward == r);
        REQUIRE(child.parent == 0);
        REQUIRE(mcts.tree.size() == expected_treesize++);
        if (mcts.is_fully_expanded(0)) {
            REQUIRE(child.t == mcts.root().t + 1);
        } else if (child.taken == null_token) {
            REQUIRE(child.t == mcts.root().t + 1);
        } else {
            REQUIRE(child.t == mcts.root().t);
        }
    }
}

TEST_CASE("tree_policy") {
    EnvGraph g(11, .1, 10);
    MCTS mcts(g);
    REQUIRE(mcts.root().children.size() == 0);
    REQUIRE(mcts.tree.size() == 1);
    auto n_children_added = 1;
    int expected_treesize = 2;
    while (!mcts.is_fully_expanded(0)) {
        mcts.tree_policy();
        REQUIRE(mcts.root().children.size() == n_children_added++);
        REQUIRE(mcts.tree.size() == expected_treesize++);
    }
}


TEST_CASE("backup") {

    EnvGraph g(10, .1, 10);
    MCTS mcts(g);
    auto& root = mcts.tree[0];
    mcts.backup(0, 10);
    REQUIRE(root.visits == 2);
    REQUIRE(root.reward == 10);
    REQUIRE(root.t == 0);
    auto child_id = mcts.expand(0);
    
    if (mcts.tree[child_id].taken == null_token) {
        REQUIRE(mcts.tree[1].reward == 0);
        mcts.backup(1, 17);
        REQUIRE(mcts.tree[1].reward == 17);
        REQUIRE(mcts.tree[0].reward == 27);
    } else {
        REQUIRE(mcts.tree[1].reward == 2);
        mcts.backup(1, 17);
        REQUIRE(mcts.tree[1].reward == 19);
        REQUIRE(mcts.tree[0].reward == 27);
    }
}


TEST_CASE("rollout", "MCTS") {
    EnvGraph g(5, .1, 10);
    MCTS mcts(g, 5, 1, 10, 10, 5);
    double r1 = mcts.rollout(0);
    double r2 = mcts.rollout(0);
    //REQUIRE(r1 != r2);
}

TEST_CASE("is_fully_expanded", "MCTS") {
    EnvGraph g(0, .1, 1);
    MCTS mcts(g, 5);
    auto actions = mcts.get_actions(0);
    REQUIRE(actions.size() == 1);
    REQUIRE(actions[0] == null_token);
    REQUIRE(!mcts.is_fully_expanded(0));
}

TEST_CASE("simulating populates root's children", "MCTS") {
    EnvGraph g(5, .1, 5);
    MCTS mcts(g, 5, 0.7071, 1, 10, 5); // env, t, sc, max_depth, num_sims, num_rolls
    mcts.simulate();
    REQUIRE(!mcts.tree[0].children.empty());
    
}


TEST_CASE("run and check objs", "MCTS") {
    EnvGraph env(5, .1, 10);
    Solution mcts_sol = MCTS::run(env, 1, 10, 10, 1); // env, sc, max_depth, num_sims, num_rolls
    Solution opt_sol = solve_optimally(env);
    REQUIRE(mcts_sol.obj <= opt_sol.obj);
    Solution greedy_sol = solve_greedily(env);
    REQUIRE(greedy_sol.obj <= opt_sol.obj);
    
    std::cout << "MCTS obj: " << mcts_sol.obj << "\n";
    std::cout << "OPT obj: " << opt_sol.obj << "\n";
    std::cout << "Greedy obj: " << greedy_sol.obj << "\n";

}







