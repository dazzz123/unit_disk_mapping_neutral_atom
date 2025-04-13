]add UnitDiskMapping
import Pkg; Pkg.add("GenericTensorNetworks")
import Pkg; Pkg.add("Graphs")
import Pkg; Pkg.add("LuxorGraphPlot")
using UnitDiskMapping, Graphs, GenericTensorNetworks, LinearAlgebra
using UnitDiskMapping.LuxorGraphPlot.Luxor, LuxorGraphPlot
n = 5
J = triu(randn(n, n), 1); J += J'
h = randn(n) 
Q = J
for i in 1:n
    Q[i, i] = h[i]
end

println(Q)

Q = [
     1   -1   2  -2;
    -1    5   3   4;
     2    3  -3  -5;
    -2    4  -5   4
]



# Extract h (diagonal)
h = diag(Q)

# Set diagonal to 0 for J
J = copy(Q)
for i in 1:size(J,1)
    J[i,i] = 0
end

println("J = ")
println(J)
println("h = ")
println(h)

qubo = UnitDiskMapping.map_qubo(J, h);

qubo_graph, qubo_weights = UnitDiskMapping.graph_and_weights(qubo.grid_graph)

 print(qubo.grid_graph.nodes)

show(qubo)
qubo_mapped_solution1 = collect(Int, solve(GenericTensorNetwork(IndependentSet(qubo_graph, qubo_weights)), SingleConfigMax())[].c.data)

qubo_mapped_solution2=[1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]

show_config(qubo.grid_graph, qubo_mapped_solution1)

show_config(qubo.grid_graph, qubo_mapped_solution2)

a1=map_config_back(qubo, collect(Int, qubo_mapped_solution1))

a2=map_config_back(qubo, collect(Int, qubo_mapped_solution2))

transpose(a1) * Q * a1
transpose(a2) * Q * a2










