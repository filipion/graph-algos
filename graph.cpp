#include <fstream>
#include <iostream>
#include <vector>
#include <stack>
#include <queue>
#include <deque>
#include <algorithm>
#include <functional>
#include <ctime>
#include <cassert>

using namespace std;
const int infty = 2*1e5;

const int max_matrix = 31;
using Adj_Matrix = array<array<int, max_matrix>, max_matrix>;

// Elements of the adjacency list we use. 
// When edge(v, d) is in adj[u], v represents the outvertex and d the weight of the edge from u to v.
struct Edge{
	int vertex;
	int weight;
	Edge(int v, int w){
		vertex = v, weight = w;
	}
};


struct SymmetricEdge{
	int from;
	int to;
	int id;
	int weight;
	SymmetricEdge(int from, int to, int id, int weight=1): from{from}, to{to}, id{id}, weight{weight} {}
};


// Weighted (ordered) graph
class Graph{
	private:
		int size;
		vector<vector<Edge>> adj;
		vector<vector<SymmetricEdge>> sym_adj;

		void dfs_recursive(int v, vector<bool>& vis, function<void(int)>, function<void(int)>);
		void bfs_recursive(vector<bool>& vis, queue<int>& q);

	public:
		Graph(int sz) :size {sz}, adj {vector<vector<Edge>>(sz + 1)}, sym_adj {vector<vector<SymmetricEdge>>(sz + 1)} {}

		void push_vertex();
		void pop_vertex();
		void add_edge(int v1, int v2, int weight);
		void add_symmetric_edge(int v1, int v2, int id, int weight);
		void print();

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
		void euler_helper(int node, vector<int>& vis, vector<int>& cycle);
};


// Basic graph algorithms and manipulating the data structure:


void Graph::add_edge(int v1, int v2, int weight=1){
	// Adds an edge from v1 to v2 in the graph with weight "weight"
	adj[v1].push_back(Edge(v2, weight));
}


void Graph::add_symmetric_edge(int from, int to, int id, int weight=1){
	SymmetricEdge e1(from, to, id, weight);
	SymmetricEdge e2(to, from, id, weight);
	sym_adj[from].push_back(e1);
	sym_adj[to].push_back(e2);
}


void Graph::print(){
	cout << "Graph on " << size << " vertices with edges:";

	for(int i=1; i <= size; ++i){
		cout << i << ":";
		for(auto edge: adj[i])
			printf("(to: %d, weight: %d) ", edge.vertex, edge.weight);
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

	for(auto e: adj[v])
		if(!vis[e.vertex])
			dfs_recursive(e.vertex, vis, discovery_action, finish_action);

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

	for(auto e: adj[v])
		if(!vis[e.vertex]){
			q.push(e.vertex);
			vis[e.vertex] = 1;
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
		for(auto e: adj[v])
			rg.add_edge(e.vertex, v);

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
		int v = pq.top().first;
		pq.pop();
		if(vis[v])
			continue;
		vis[v] = 1;

		for(auto edge: adj[v])
			if(distance[edge.vertex] > distance[v] + edge.weight){
				distance[edge.vertex] = distance[v] + edge.weight;
				pq.push(make_pair(edge.vertex, distance[edge.vertex]));
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
				if(distance[edge.vertex] > distance[v] + edge.weight)
					distance[edge.vertex] = distance[v] + edge.weight;
	}

	// If executing the body of the previous for once again would decrease distances, then a negative cycle exists.
	for(int v = 1; v <= size; ++v)
		for(auto edge: adj[v])
			if(distance[edge.vertex] > distance[v] + edge.weight)
				distance = {};

	return distance;
}


Adj_Matrix all_shortest_paths_floyd_warshall(int n, Adj_Matrix M){
	int dp[max_matrix][max_matrix][max_matrix];
	Adj_Matrix ans;

	for(int i = 1; i <= n; ++i)
		for(int j = 1; j <= n; ++j)
			dp[i][j][0] = M[i][j];

	for(int k = 1; k <= n; ++k)
		for(int i = 1; i <= n; ++i)
			for(int j = 1; j <= n; ++j)
				dp[i][j][k] = min(dp[i][j][k - 1], dp[i][k][k - 1] + dp[k][j][k - 1]);

	for(int i = 1; i <= n; ++i)
		for(int j = 1; j <= n; ++j)
			ans[i][j] = dp[i][j][n];

	return ans;
}


vector<vector<int>> Graph::all_shortest_paths_johnsons(){
	/*
	Johnson's algorithm for all shortest paths. Returns a table all_paths such that all_paths[i][j] is the shortest
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
			reweighted_g.add_edge(i, edge.vertex, h[edge.vertex] - h[i] + edge.weight);
			
	vector<vector<int>> all_paths(size + 1);

	for(int i = 1; i <= size; ++i){
		all_paths[i] = reweighted_g.shortest_paths_dijkstra(i);
		for(int j = 1; j <= size; ++j)
				all_paths[i][j] += (h[i] - h[j]); // converting to the original weights
	}

	return all_paths;
}


// Primm algorithm for minimal spanning tree. The code here is just dijkstra with a slightly different condition.
vector<int> Graph::minimal_spanning_tree(int source=1){
	vector<bool> vis(size + 1);
	vector<int> key(size + 1);
	vector<int> parent(size + 1);
	int v, d;
	int total = 0;

	for(int i = 1; i <= size; ++i)
		key[i] = infty;
	key[source] = 0;

	// The priority queue holds edges to the undiscovered parts of the graph, keyed by distance to the discovered part
	auto cmp = [&](pair<int, int> x, pair<int, int> y){ return x.second > y.second; };
	priority_queue <pair<int,int>, vector<pair<int,int>>, decltype(cmp)> pq(cmp);
	pq.push(make_pair(source, 0));

	while(!pq.empty()){
		v = pq.top().first;
		d = pq.top().second;
		pq.pop();
		if(vis[v])
			continue;
		vis[v] = 1;
		total += d; 

		for(auto edge: adj[v])
			if(key[edge.vertex] > edge.weight && !vis[edge.vertex]){
				key[edge.vertex] = edge.weight;
				parent[edge.vertex] = v;
				pq.push(make_pair(edge.vertex, key[edge.vertex]));
			}
	}

	parent[0] = total;
	return parent;
}


Graph random_graph(int size){
	Graph g = Graph(size);
	for(int i = 1; i <= size; ++i)
		for(int j = 1; j <= size; ++j){
			if(rand() % 3 == 1)
				g.add_edge(i, j, rand() % 10 + 1);
		}

	return g;
}


vector<int> Graph::eulerian_cycle(int m){
	// Return {-1} for non eulerian graphs.
	for(int i = 1; i <= size; ++i)
		if(sym_adj[i].size() & 1)
			return vector<int> {-1};

	vector<int> cycle;
	vector<int> vis(m + 1);
	
	euler_helper(1, vis, cycle);
	return cycle;
}


void Graph::euler_helper(int node, vector<int>& vis, vector<int>& cycle){

	while(!sym_adj[node].empty()){
		SymmetricEdge edge = sym_adj[node].back();
		sym_adj[node].pop_back();


		if(!vis[edge.id]){
			vis[edge.id] = 1;
			euler_helper(edge.to, vis, cycle);
		}
	}

	cycle.push_back(node);
}


int main(){
	
	srand(time(0));
	ifstream fin;
	ofstream fout;
	fin.open("royfloyd.in");
	fout.open("royfloyd.out");

	int n, m, u, v, weight;
	fin >> n;
	Graph g(n);

	for(int i=1; i <= n; ++i)
		for(int j=1; j <= n; ++j){
			fin >> weight;
			if(weight || i == j)
				g.add_edge(i, j, weight);
			else
				g.add_edge(i, j, infty);
		}

	vector<vector<int>> all_paths = g.all_shortest_paths_johnsons();

	for(int i=1; i <= n; ++i){
		for(int j=1; j <= n; ++j){
			fout << ((all_paths[i][j] >= infty) ? 0 : all_paths[i][j]) << " ";
		}
		fout << "\n";
	}
}