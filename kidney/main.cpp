//
//  main.cpp
//  kidney
//
//  Created by Vitor Baisi Hadad on 10/26/17.
//  Copyright © 2017 halflearned. All rights reserved.
//

#include <stdio.h>
#include <iostream>
#include "solver.hpp"
#include "environment.hpp"
#include "mcts.hpp"

int main(int argc, char *argv[]) {
    
    double entry_rate = atof(argv[1]);
    double death_rate = atof(argv[2]);
    int max_time = atoi(argv[3]);
    double scaling = atof(argv[4]);
    int max_depth = atoi(argv[5]);
    int num_sims = atoi(argv[6]);
    int num_rolls = atoi(argv[7]);
    
    std::cout << "Entry rate: " << entry_rate << "\n";
    std::cout << "Death rate: " << death_rate << "\n";
    std::cout << "Max time: " << max_time << "\n";
    std::cout << "Scaling: " << scaling << "\n";
    std::cout << "Max depth: " << max_depth << "\n";
    std::cout << "Num sims: " << num_sims << "\n";
    std::cout << "Num rolls: " << num_rolls << "\n";
    
    auto env = EnvGraph(entry_rate, death_rate, max_time);
    Solution opt_sol = solve_optimally(env);
    Solution greedy_sol = solve_greedily(env);
    Solution mcts_sol = MCTS::run(env, scaling, max_depth, num_sims, num_rolls);
    std::cout << "OPT obj: " << opt_sol.obj << "\n";
    std::cout << "Greedy obj: " << greedy_sol.obj << "\n";
    std::cout << "MCTS obj: " << mcts_sol.obj << "\n";
}

