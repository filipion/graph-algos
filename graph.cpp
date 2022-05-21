#include <fstream>
#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <functional>
using namespace std;
vector<vector<int>> SCC;
int ANS;


class Graph{
	private:
		int size;
		vector<vector<int>> adj;

		void dfs_helper(int v, vector<bool>& vis, function<void(int)>, function<void(int)>);
		void bfs_helper(vector<bool>& vis, queue<int>& q);

	public:
		Graph(int sz) :size {sz}, adj {vector<vector<int>>(sz + 1)} {}

		void add_edge(int v1, int v2);
		void add_edge(int v1, int v2, int weight);
		void print();

		void dfs(const vector<int>& ordered_vertices, function<void(int)>, function<void(int)>);
		void bfs(int start_vertex);

		Graph reverse_graph();
		void scc_helper(int v, vector<bool>& vis, vector<int>& order_for_scc);
		vector<int> compute_order_scc();
		void strongly_connected_components();

};


//Basic utilities, add edge, add weighted edge, print to stdout etc.
void Graph::add_edge(int v1, int v2){
	adj[v1].push_back(v2);
}


void Graph::print(){
	for(int i=1; i <= size; ++i){
		cout << i << ":";
		for(auto u: adj[i])
			cout << " " << u;
		cout << "\n";
	}
}


// The recursive dfs traversal algorithm.
// Iterates through the vector of ordered vertices and calls dfs whenever it finds a new component. 
// We initialize a vector vis of visits locally and pass it by reference to prevent useless copying.
void Graph::dfs(const vector<int>& ordered_vertices = {}, function<void(int)> discovery_action = [](int){}, function<void(int)> finish_action = [](int){}){

	vector<bool> vis(size);

	if(!ordered_vertices.empty()){ // case of the second dfs of strongly connected components
		int count_scc = 0;
		for(auto v: ordered_vertices){
			if(!vis[v]){
				++count_scc;
				SCC.push_back(vector<int> {});
				dfs_helper(v, vis, discovery_action, finish_action);
			}
		}

		ANS = count_scc;
	}
	else{ // default DFS
		for(int v = 1; v <= size; ++v)
			if(!vis[v])
				dfs_helper(v, vis, discovery_action, finish_action);
	}
}


// Basic dfs recursive function.
void Graph::dfs_helper(int v, vector<bool>& vis, function<void(int)> discovery_action, function<void(int)> finish_action){
	vis[v] = 1;
	discovery_action(v);

	for(auto u: adj[v])
		if(!vis[u])
			dfs_helper(u, vis, discovery_action, finish_action);
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

	for(auto u: adj[v]) //extend queue
		if(!vis[u]){
			q.push(u);
			vis[u] = 1;
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
	vector<int> order_for_scc;
	// Because DFS takes functions as parameters we do not have to implement it again
	dfs({}, [](int){}, [&order_for_scc](int v){order_for_scc.push_back(v);}); 
	reverse(order_for_scc.begin(), order_for_scc.end());

	Graph g_reversed = this->reverse_graph();

	vector<bool> vis(size);

	for(auto v: order_for_scc){
		if(!vis[v]){
			// we DFS traverse again. Each call of dfs fills out a SCC
			SCC.push_back(vector<int> {});
			g_reversed.dfs_helper(v, vis, [](int v){SCC.back().push_back(v);}, [](int v){}); 
		}
	}
}


// Reverse a graph
Graph Graph::reverse_graph(){
	Graph rg(size);

	for(int v = 1; v <= size; ++v)
		for(auto u: adj[v])
			rg.add_edge(u, v);

	return rg;
}



int main(){
	ifstream fin;
	ofstream fout;
	fin.open("ctc.in");
	fout.open("ctc.out"); 

	
	int n, m, u, v;
	fin >> n >> m;
	Graph g(n);

	for(int i=1; i <= m; ++i){
		fin >> u >> v;
		g.add_edge(u, v);
	}

	
	g.strongly_connected_components();
	fout << SCC.size() << "\n";
	for(auto& component: SCC){
		for(auto v: component)
			fout << v << " ";
		fout << "\n";
	}

}