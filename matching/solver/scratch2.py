import gurobi as gb

m = gb.Model()
x = m.addVar(vtype=gb.GRB.BINARY, obj=1234)
m.setObjective(x, gb.GRB.MAXIMIZE)
m.update()
m.optimize()