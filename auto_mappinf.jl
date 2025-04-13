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

qubo = UnitDiskMapping.map_qubo(J, h);

qubo_graph, qubo_weights = UnitDiskMapping.graph_and_weights(qubo.grid_graph)

 result=qubo.grid_graph.nodes

show(qubo)
using JSON
open("julia_output.json", "w") do f
    JSON.print(f, result)
end
