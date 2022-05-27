#include <fstream>
#include <iostream>
#include <vector>
#include <queue>
#include <functional>

using namespace std;
const int infty = 2*1e9;

const int max_matrix = 31;
using Adj_Matrix = array<array<int, max_matrix>, max_matrix>;

// Elements of the adjacency lists we will use. 
// When edge(v, d) is in adj[u], v represents the outvertex and d the weight of the edge from u to v.
struct Edge{
	int to;
	int weight;
	int tag; // a tag for the edges needed for example when computing an eulerian cycle
	Edge(int destination, int w, int tag){
		to = destination, weight = w, tag = tag;
	}
};


// Weighted (ordered) graph
class Graph{
	private:
		int size; // the number of vertices  
		int num_edges;              
		vector<vector<Edge>> adj; //holds the adjacency list representation of the graph

		void dfs_recursive(int v, vector<bool>& vis, function<void(int)>, function<void(int)>);
		void bfs_recursive(vector<bool>& vis, queue<int>& q);
		void euler_helper(int node, vector<bool>& vis, vector<int>& cycle);

	public:
		Graph(int sz) :size {sz}, adj {vector<vector<Edge>>(sz + 1)} {}

		void push_vertex();
		void pop_vertex();
		void add_edge(int v1, int v2, int weight);
		void add_symmetric_edge(int v1, int v2, int weight);
		void print();
		int get_num_edges();

		void dfs(function<void(int)>, function<void(int)>);
		void bfs(int start_vertex);

		Graph reversed_graph();
		vector<vector<int>> strongly_connected_components();

		vector<int> minimal_spanning_tree(int source);

		vector<int> shortest_paths_dijkstra(int source);
		vector<int> shortest_paths_bellman_ford(int source);

		int** all_shortest_paths(int** adj_matrix);
		vector<vector<int>> all_shortest_paths_johnsons();

		vector<int> eulerian_cycle(int number_edges);
};


// Basic graph algorithms and manipulating the data structure:


void Graph::add_edge(int v1, int v2, int weight=1){
	// Adds an edge from v1 to v2 in the graph with weight "weight"
	int id = ++num_edges;
	Edge e = Edge(v2, weight, id);
	adj[v1].push_back(e);
	num_edges = num_edges + 1;
}


void Graph::add_symmetric_edge(int v1, int v2, int weight=1){
	// Adds an edge from v1 to v2 and the opposite edge, both with the same tag.
	// Use this for unoriented graphs.
	int id = ++num_edges;
	Edge e1 = Edge(v2, weight, id);
	Edge e2 = Edge(v1, weight, id);
	adj[v1].push_back(e1);
	adj[v2].push_back(e2);
}


int Graph::get_num_edges(){
	return num_edges;
}


void Graph::print(){
	cout << "Graph on " << size << " vertices with edges:";

	for(int i=1; i <= size; ++i){
		cout << i << ":";
		for(auto edge: adj[i])
			printf("(to: %d, weight: %d) ", edge.to, edge.weight);
		cout << "\n";
	}
}


void Graph::push_vertex(){
	vector<Edge> neighbor_list;
	adj.push_back(neighbor_list);
	++size;
}


void Graph::pop_vertex(){
	adj.pop_back();
	--size;
}



void Graph::dfs(function<void(int)> discovery_action = [](int){}, function<void(int)> finish_action = [](int){}){
	/* The recursive depth-first traversal algorithm for a possibly unconnected graph.
 	Iterates through the vector of ordered vertices and calls dfs whenever it finds a new component.
	*/
	vector<bool> vis(size);

	for(int v = 1; v <= size; ++v)
		if(!vis[v])
			dfs_recursive(v, vis, discovery_action, finish_action);
}


void Graph::dfs_recursive(int v, vector<bool>& vis, function<void(int v)> discovery_action = [](int){}, function<void(int v)> finish_action = [](int){}){
	/* Basic depth-first search recursive function. Supports passing functional callbacks to execute at vertex discovery or finish.
	Call this directly instead of Graph::dfs if you need to depth first search the graph in a custom order
	*/
	vis[v] = 1;
	discovery_action(v);

	for(auto edge: adj[v])
		if(!vis[edge.to])
			dfs_recursive(edge.to, vis, discovery_action, finish_action);

	finish_action(v);
}


void Graph::bfs(int start_vertex){
	/* Breadth-first search traversal algorithm.
	*/
	vector<bool> vis(size);
	queue<int> q;

	q.push(start_vertex);
	vis[start_vertex] = 1;

	bfs_recursive(vis, q);
}


void Graph::bfs_recursive(vector<bool>& vis, queue<int>& q){
	int v = q.front();

	for(auto edge: adj[v])
		if(!vis[edge.to]){
			q.push(edge.to);
			vis[edge.to] = 1;
		}

	q.pop(); 
	if(!q.empty())
		bfs_recursive(vis, q);
}


// Connectivity algorithms


vector<vector<int>> Graph::strongly_connected_components(){
	/* Strongly connected components with Kosaraju's algorithm. 
    Uses a slightly modified DFS, that builds up a vector of the vertices in reverse order of their finishing times.
    This ordering satisfies the property that the subsequence of the first occurences of all SCCs is a topologicaly sorted subsequence.
	Because our DFS implementation takes lambdas as parameters we do not have to implement it again.
	*/
	vector<int> order_for_scc;
	vector<vector<int>> SCC;
	dfs([](int){}, [&order_for_scc](int v){order_for_scc.push_back(v);}); //construct the ordering

	Graph g_reversed = this->reversed_graph();
	
	vector<bool> vis(size);
	for(auto it = order_for_scc.rbegin(); it != order_for_scc.rend(); ++it){
		int v = *it;
		if(!vis[v]){
			// we DFS traverse again. Each call of dfs fills out a SCC
			SCC.push_back(vector<int> {});
			g_reversed.dfs_recursive(v, vis, [&](int v){SCC.back().push_back(v);}); 
		}
	}

	return SCC;
}


Graph Graph::reversed_graph(){
	/* Returns a new graph with reversed edges
	*/
	Graph rg(size);

	for(int v = 1; v <= size; ++v)
		for(auto edge: adj[v])
			rg.add_edge(edge.to, v);

	return rg;
}


// Shortest path algorithms


vector<int> Graph::shortest_paths_dijkstra(int source){
	/* Dijkstra's algorithm for shortest paths from a source. Requires positive weights.
	Because pq is pushed to at most E times, where E is the number of edges, pq.size is always <= E.
	The number of operations is thus O(E * log E) which equals O(E * log V) as E < V^2vector<bool> vis(size + 1);
	*/
	vector<bool> vis(size + 1);
	vector<int> distance(size + 1);
	for(auto& d: distance)
		d = infty;
	distance[source] = 0;

	// The priority queue holds edges to the undiscovered parts of the graph.
	auto cmp = [&](pair<int, int> x, pair<int, int> y){ return x.second > y.second; };
	priority_queue <pair<int,int>, vector<pair<int,int>>, decltype(cmp)> pq(cmp);
	pq.push(make_pair(source, 0));

	while(!pq.empty()){
		int current_vertex = pq.top().first;
		pq.pop();
		if(vis[current_vertex])
			continue;
		vis[current_vertex] = 1;

		for(auto edge: adj[current_vertex])
			if(distance[edge.to] > distance[current_vertex] + edge.weight){
				distance[edge.to] = distance[current_vertex] + edge.weight;
				pq.push(make_pair(edge.to, distance[edge.to]));
			}
	}

	return distance;
}


vector<int> Graph::shortest_paths_bellman_ford(int source){
	/* Bellman Ford shortest path algorithm.
    If it detects a negative cycle, the algorithm returns an empty vector of distances.
	*/
	vector<bool> vis(size + 1);
	vector<int> distance(size + 1);
	for(auto& d: distance)
		d = infty;
	distance[source] = 0;

	for(int i = 1; i <= size - 1; ++i){// repeat |V| - 1 times
		// Body of the loop traverses all edges once and relaxes the distance vector.
		for(int v = 1; v <= size; ++v)
			for(auto edge: adj[v])
				if(distance[edge.to] > distance[v] + edge.weight)
					distance[edge.to] = distance[v] + edge.weight;
	}

	// If executing the body of the previous for once again would decrease distances, then a negative cycle exists.
	for(int v = 1; v <= size; ++v)
		for(auto edge: adj[v])
			if(distance[edge.to] > distance[v] + edge.weight)
				distance = {};

	return distance;
}


Adj_Matrix all_shortest_paths_floyd_warshall(int n, Adj_Matrix M){
	/* Floyd-Warshall algorithm for al shortest paths. Does not work with adjacency lists directly, you need to pass a max_size
	and the Adj_Matrix. 
	*/

	// dp is our dynamic programming data structure.
	// dp[i][j][k] := length of the shortest path from i to j whose intermediary vertices are all included in [1..k] 
	int dp[max_matrix][max_matrix][max_matrix];
	Adj_Matrix ans;

	for(int i = 1; i <= n; ++i)
		for(int j = 1; j <= n; ++j)
			dp[i][j][0] = M[i][j];

	for(int k = 1; k <= n; ++k)
		for(int i = 1; i <= n; ++i)
			for(int j = 1; j <= n; ++j)
				dp[i][j][k] = min(dp[i][j][k - 1],  // case 1: the optimal path that does not include k
							      dp[i][k][k - 1] + dp[k][j][k - 1]); // case 2: the optimal path that includes k is formed of two portions that di not include it

	for(int i = 1; i <= n; ++i)
		for(int j = 1; j <= n; ++j)
			ans[i][j] = dp[i][j][n];

	return ans;
}


vector<vector<int>> Graph::all_shortest_paths_johnsons(){
	/* Johnson's algorithm for all shortest paths. Returns a table all_paths such that all_paths[i][j] is the shortest
	possible distance of a route from i to j.
	*/
	this->push_vertex(); //add a virtual vertex to compute reweightings
	for(int i = 1; i < size; ++i)
		this->add_edge(size, i, 0);

	vector<int> h;

	h = this->shortest_paths_bellman_ford(size);
	this->pop_vertex();

	Graph reweighted_g(size);

	for(int i = 1; i <= size; ++i)
		for(auto edge: adj[i])
			reweighted_g.add_edge(i, edge.to, h[edge.to] - h[i] + edge.weight);
			
	// We apply Dijkstra repeatedly to get the distances in the reweighted graph. Adding h[i] - h[j] gives back the original dists.	
	vector<vector<int>> all_paths(size + 1);

	for(int i = 1; i <= size; ++i){
		all_paths[i] = reweighted_g.shortest_paths_dijkstra(i);
		for(int j = 1; j <= size; ++j)
				all_paths[i][j] += (h[i] - h[j]); // converting to the original weights
	}

	return all_paths;
}


vector<int> Graph::minimal_spanning_tree(int source=1){
	/* Primm algorithm for minimal spanning tree. Returns the a vector of size V + 1. 
	For i from 2 to n, MST[i] is the parent of i in the MST.
	MST[1] is not used. MST[0] returns the total cost of the spanning tree.
	*/
	vector<bool> vis(size + 1);
	vector<int> key(size + 1);
	vector<int> parent(size + 1);
	int current_vertex, weight_to_here;
	int total = 0;

	for(int i = 1; i <= size; ++i)
		key[i] = infty;
	key[source] = 0;

	// The priority queue holds edges to the undiscovered parts of the graph, keyed by distance to the discovered part
	auto cmp = [&](pair<int, int> x, pair<int, int> y){ return x.second > y.second; };
	priority_queue <pair<int,int>, vector<pair<int,int>>, decltype(cmp)> pq(cmp);
	pq.push(make_pair(source, 0));

	while(!pq.empty()){
		current_vertex = pq.top().first;
		weight_to_here = pq.top().second;
		pq.pop();
		if(vis[current_vertex])
			continue;
		vis[current_vertex] = 1;
		total += weight_to_here; 

		for(auto edge: adj[current_vertex])
			if(key[edge.to] > edge.weight && !vis[edge.to]){
				key[edge.to] = edge.weight;
				parent[edge.to] = current_vertex;
				pq.push(make_pair(edge.to, key[edge.to]));
			}
	}

	parent[0] = total;
	return parent;
}


Graph random_graph(int size, int sparsity){
	// Probability for the edge (i, j) to exist in the graph is 1/sparsity. Weights are nitialized to random digits.
	Graph g(size);
	for(int i = 1; i <= size; ++i)
		for(int j = 1; j <= size; ++j){
			if(rand() % sparsity == 1)
				g.add_edge(i, j, rand() % 10 + 1);
		}

	return g;
}


vector<int> Graph::eulerian_cycle(int num_edges){
	/* Returns an eulerian cycle for graphs that admit such cycles, and the vector {-1} for graphs that do not.
	*/
	for(int i = 1; i <= size; ++i) // check if the graph is eulerian
		if(adj[i].size() & 1)
			return vector<int> {-1};

	vector<int> cycle;
	vector<bool> vis(num_edges + 1);
	
	euler_helper(1, vis, cycle);
	return cycle;
}


void Graph::euler_helper(int node, vector<bool>& vis, vector<int>& cycle){

	while(!adj[node].empty()){
		Edge edge = adj[node].back();
		adj[node].pop_back();


		if(!vis[edge.tag]){
			vis[edge.tag] = 1;
			euler_helper(edge.to, vis, cycle);
		}
	}

	cycle.push_back(node);
}


Graph read_graph_from_file(string f_name, bool unoriented=false){
	/* Reads a graph from the supplied filename. 
	Format: - First line shoud consist of two integers V and E, the number of vertices and edges respectively.
			- Each of the following lines should consist of 3 integers, u, v, and w := there exists and edge (u, v) of weight w
	*/
	ifstream fin;
	fin.open(f_name);
	int num_vertices, num_edges;
	fin >> num_vertices;
	Graph g(num_vertices);

	int u, v, w;
	for(int i = 1; i <= num_edges; ++i){
		fin >> u >> v;
		if(unoriented)
			g.add_symmetric_edge(u, v, w);
		else
			g.add_edge(u, v, w);
	}

	return g;
}

int main(){}