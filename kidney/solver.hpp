//
//  solver.hpp
//  kidney
//
//  Created by Vitor Baisi Hadad on 10/23/17.
//  Copyright © 2017 halflearned. All rights reserved.
//

#ifndef solver_h
#define solver_h

#include <gurobi_c++.h>
#include <set>
#include <memory>


using cycles = std::vector<std::vector<unsigned long>>;

struct Solution {
    Solution() = default;
    Solution(int o, std::set<unsigned long> c): obj(o), chosen(c) {};
    int obj;
    std::set<unsigned long> chosen;
};


class Solver {
public:
    Solver(cycles c, int n, int mcl = 2):
    cs(c), num_vertices(n), max_cycle_length(mcl) {};
    int max_cycle_length;
    int num_vertices;
    cycles cs;
    Solution solve();
private:
    Solution parse_solution(const GRBModel& model) const;
};


Solution Solver::parse_solution(const GRBModel& model) const {
    auto num_vars = model.get(GRB_IntAttr_NumVars);
    auto vp = std::unique_ptr<GRBVar>(model.getVars());
    std::vector<GRBVar> cycle_variables(vp.get(), vp.get() + num_vars*sizeof(GRBVar));
    std::set<unsigned long> values;
    double obj = model.get(GRB_DoubleAttr_ObjVal);
    
    for (int i = 0; i < cs.size(); ++i) {
        if (cycle_variables[i].get(GRB_DoubleAttr_X)) {
            for (auto&v: cs[i])
                values.insert(v);
        }
    }
    return Solution(obj, values);
}


Solution Solver::solve() {
    if (cs.size() == 0) {
        return Solution(0, {});
    }
    
    // Setup
    GRBEnv grbenv = GRBEnv();
    GRBModel model = GRBModel(grbenv);
    model.set(GRB_IntParam_OutputFlag, false);
    std::vector<GRBVar> cycle_variables(cs.size());
    std::vector<GRBLinExpr> vertex_constraints(num_vertices);
    GRBLinExpr obj;

    // Constraints and objective
    for (auto& c: cs) {
        auto cyc_var = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
        obj += c.size() * cyc_var; // TODO: weights
        for (auto& v: c)
            vertex_constraints[v] += cyc_var;
    }
    for (auto& constr: vertex_constraints) {
        model.addConstr(constr, GRB_LESS_EQUAL, 1.0, "");
    }

    // Optimize!
    model.update();
    model.setObjective(obj, GRB_MAXIMIZE);
    model.optimize();
    
    // Parse
    return parse_solution(model);
}











#endif /* solver_h */

