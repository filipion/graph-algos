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
vector<vector<int>> SCC;
const int infty = 2*1e9;
const int max_matrix = 31;
int DEBUG;
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


// Weighted graph class for the purpose of demonstrating graph algorithms
class Graph{
	private:
		int size;
		vector<vector<Edge>> adj;
		vector<vector<SymmetricEdge>> sym_adj;

		void dfs_helper(int v, vector<bool>& vis, function<void(int)>, function<void(int)>);
		void bfs_helper(vector<bool>& vis, queue<int>& q);

	public:
		Graph(int sz) :size {sz}, adj {vector<vector<Edge>>(sz + 1)}, sym_adj {vector<vector<SymmetricEdge>>(sz + 1)} {}

		void add_edge(int v1, int v2, int weight);
		void add_symmetric_edge(int v1, int v2, int id, int weight);
		void print();

		void dfs(function<void(int)>, function<void(int)>);
		void bfs(int start_vertex);

		Graph reverse_graph();
		void strongly_connected_components();

		vector<int> minimal_spanning_tree(int source);

		vector<int> shortest_paths_dijkstra(int source);
		vector<int> shortest_paths_bellman_ford(int source);

		int** all_shortest_paths(int** adj_matrix);

		vector<int> eulerian_cycle(int number_edges);
		void euler_helper(int node, vector<int>& vis, vector<int>& cycle);
};


//Basic utilities, add weighted edge, print to stdout etc.
void Graph::add_edge(int v1, int v2, int weight=1){
	adj[v1].push_back(Edge(v2, weight));
}


void Graph::add_symmetric_edge(int from, int to, int id, int weight=1){
	SymmetricEdge e1(from, to, id, weight);
	SymmetricEdge e2(to, from, id, weight);
	sym_adj[from].push_back(e1);
	sym_adj[to].push_back(e2);
}


void Graph::print(){
	for(int i=1; i <= size; ++i){
		cout << i << ":";
		for(auto edge: adj[i])
			printf("(%d, %d) ", edge.vertex, edge.weight);
		cout << "\n";
	}
}


// The recursive dfs traversal algorithm.
// Iterates through the vector of ordered vertices and calls dfs whenever it finds a new component. 
// We initialize a vector vis of visits locally and pass it by reference to prevent useless copying.
void Graph::dfs(function<void(int)> discovery_action = [](int){}, function<void(int)> finish_action = [](int){}){
	vector<bool> vis(size);

	for(int v = 1; v <= size; ++v)
		if(!vis[v])
			dfs_helper(v, vis, discovery_action, finish_action);
}


// Basic dfs recursive function. Supports passing functional callbacks to execute at vertex discovery or finish.
void Graph::dfs_helper(int v, vector<bool>& vis, function<void(int v)> discovery_action = [](int){}, function<void(int v)> finish_action = [](int){}){
	vis[v] = 1;
	discovery_action(v);

	for(auto e: adj[v])
		if(!vis[e.vertex])
			dfs_helper(e.vertex, vis, discovery_action, finish_action);
	finish_action(v);
}


// Bfs traversal algorithm.
// As before we maintain a visits vector and a queue of the next vertices to visit.
void Graph::bfs(int start_vertex){
	vector<bool> vis(size);
	queue<int> q;

	q.push(start_vertex);
	vis[start_vertex] = 1;

	bfs_helper(vis, q);
}


void Graph::bfs_helper(vector<bool>& vis, queue<int>& q){
	int v = q.front(); //starting vertex

	for(auto e: adj[v]) //extend queue
		if(!vis[e.vertex]){
			q.push(e.vertex);
			vis[e.vertex] = 1;
		}

	q.pop(); 
	if(!q.empty())
		bfs_helper(vis, q);
}


// Strongly connected components. 
// Uses slightly modified DFS, that builds up a vector of the vertices in reverse order of their finishing times.
// Required by Kosaraju's algorithm for SCC. This ordering satisfies the property that the first occurences
// of all SCCs form a topologicaly sorted subsequence.
void Graph::strongly_connected_components(){
	// Because DFS takes functions as parameters we do not have to implement it again
	vector<int> order_for_scc;
	dfs([](int){}, [&order_for_scc](int v){order_for_scc.push_back(v);});

	Graph g_reversed = this->reverse_graph();
	
	vector<bool> vis(size);
	for(auto it = order_for_scc.rbegin(); it != order_for_scc.rend(); ++it){
		int v = *it;
		if(!vis[v]){
			// we DFS traverse again. Each call of dfs fills out a SCC
			SCC.push_back(vector<int> {});
			g_reversed.dfs_helper(v, vis, [](int v){SCC.back().push_back(v);}); 
		}
	}
}


// Reverse a graph
Graph Graph::reverse_graph(){
	Graph rg(size);

	for(int v = 1; v <= size; ++v)
		for(auto e: adj[v])
			rg.add_edge(e.vertex, v);

	return rg;
}


// Dijkstra's algorithm for shortest paths from a source
// STL priority queues do not support easily deleting an element, so we skip deleting the elements.
// Because pq is pushed to at most E times, where E is the number of edges, pq.size is always <= E.
// The number of operations is thus O(E * log E) which equals O(E * log V) as E < V^2
vector<int> Graph::shortest_paths_dijkstra(int source){
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


// Computes all distances from source to vertices in the graph.
// If it detects a negative cycle, the algorithm returns an empty vector of distances.
vector<int> Graph::shortest_paths_bellman_ford(int source){
	vector<bool> vis(size + 1);
	vector<int> distance(size + 1);
	for(auto& d: distance)
		d = infty;
	distance[source] = 0;

	for(int i = 1; i <= size - 1; ++i){// repeat |V| - 1 times
		// Body of the for traverses all edges once and relaxes the distance vector.
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


Adj_Matrix all_shortest_paths(int n, Adj_Matrix M){
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

ifstream fin;
ofstream fout;
int ecount = 0;


vector<int> Graph::eulerian_cycle(int m){
	
	// Return {-1} for non eulerian graphs.
	for(int i = 1; i <= size; ++i)
		if(sym_adj[i].size() & 1)
			return vector<int> {-1};


	vector<int> cycle;
	vector<int> vis(m + 1);
	int node;
	stack<int> stk;
	

	
	stk.push(1);

	while(!stk.empty()){
		node = stk.top();
		

		if(!sym_adj[node].empty()){
			SymmetricEdge edge = sym_adj[node].back();
			sym_adj[node].pop_back();

			if(!vis[edge.id]){
				vis[edge.id] = 1;
				stk.push(edge.to);
				cycle.push_back(edge.to);
				cout << edge.to;
			}
		}
		else{
			
			stk.pop();
		}
	}

	return cycle;
}


int main(){
	
	srand(time(0));
	fin.open("ciclueuler.in");
	fout.open("ciclueuler.out"); 

	
	int n, m, u, v, weight;
	fin >> n >> m;
	Graph g(n);

	for(int i=1; i <= m; ++i){
		fin >> u >> v;
		g.add_symmetric_edge(u, v, i);
	}

	vector<int> cycle;
	cycle = g.eulerian_cycle(m);


 
}