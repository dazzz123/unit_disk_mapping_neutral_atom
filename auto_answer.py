qubo_mapped_solution1 = collect(Int, solve(GenericTensorNetwork(IndependentSet(qubo_graph, qubo_weights)), SingleConfigMax())[].c.data)
data = JSON.parsefile("python_output.json")
qubo_mapped_solution2=data

show_config(qubo.grid_graph, qubo_mapped_solution1)

show_config(qubo.grid_graph, qubo_mapped_solution2)

a1=map_config_back(qubo, collect(Int, qubo_mapped_solution1))

a2=map_config_back(qubo, collect(Int, qubo_mapped_solution2))

transpose(a1) * Q * a1
transpose(a2) * Q * a2
