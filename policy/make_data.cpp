//
//  make_data.cpp
//  kidney
//
//  Created by Vitor Baisi Hadad on 11/6/17.
//  Copyright © 2017 halflearned. All rights reserved.
//

#include <stdio.h>
#include <fstream>
#include <string>
#include "solver.hpp"
#include "environment.hpp"


int main(int argc, char *argv[]) {
    
    std::cout << "Running policy" << "\n";
    
    double entry_rate = (argc > 1) ? atof(argv[1]):    5;
    double death_rate = (argc > 2) ? atof(argv[2]):   .1;
    int max_time =      (argc > 3) ? atoi(argv[3]):  100;
    int n =             (argc > 4) ? atoi(argv[4]):    1;
    std::string name =  (argc > 5) ? argv[5] : "tmp";
    
    for (int i = 0; i < n; ++i) {
        // Solve
        auto env = EnvGraph(entry_rate, death_rate, max_time);
        auto sol = solve_optimally(env);
        std::map<unsigned long, cycle> table;
        
        // Make a table
        for (auto& c: sol.matched_cycles) {
            //            const unsigned long t = *std::min_element(
            //                                      c.begin(),
            //                                      c.end(),
            //                                      [&env](int i, int j ){
            //                                          return env.nodes[i].entry < env.nodes[j].entry;
            //                                      });
            const unsigned long t = std::min(env.nodes[c[0]].entry, env.nodes[c[1]].entry);
            
            if (table.find(t) == table.end()) {
                table[t] = c;
            } else {
                std::copy(c.begin(),
                          c.end(),
                          std::back_inserter(table[t]));
            }
        }
        
        // Print solution
        std::ofstream solfile;
        solfile.open (name + "_solution.txt", std::ios::out);
        for (auto & it : table) {
            solfile << it.first << ", ";
            for (auto & v : it.second)
                solfile << v << " ";
            solfile << "\n";
        }
        solfile.close();
        
        std::ofstream envfile;
        envfile.open(name + "_environment.txt", std::ios::out);
        for (auto &v : env.nodes) {
            envfile <<
            v.id << ", " <<
            v.patient << ", " <<
            v.donor << ", " <<
            v.entry << ", " <<
            v.death << ", " <<
            v.pra << ", ";
            for (auto &o: v.out_edges) {
                envfile << o << " ";
            }
            envfile << "\n";
        }
    }
    
    
};

