#include <fstream>
#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
using namespace std;
vector<vector<int>> SCC;
int ANS;

class Graph{
	private:
		int size;
		vector<vector<int>> adj;

		void dfs_helper(int v, vector<bool>& vis);
		void bfs_helper(vector<bool>& vis, queue<int>& q);
	public:
		Graph(int sz) :size {sz}, adj {vector<vector<int>>(sz + 1)} {}

		void add_edge(int v1, int v2);
		void add_edge(int v1, int v2, int weight);
		void print();

		void dfs(const vector<int>& ordered_vertices);
		void bfs(int start_vertex);

		Graph reverse_graph();
		void scc_helper(int v, vector<bool>& vis, vector<int>& order_for_scc);
		void compute_order_scc(vector<int>&);
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
void Graph::dfs(const vector<int>& ordered_vertices = {}){
	vector<bool> vis(size);

	if(!ordered_vertices.empty()){ // case of the second dfs of strongly connected components
		int count_scc = 0;
		for(auto v: ordered_vertices){
			if(!vis[v]){
				++count_scc;
				SCC.push_back(vector<int> {});
				dfs_helper(v, vis);
			}
		}

		ANS = count_scc;
	}
	else{ // default DFS
		for(int i = 1; i <= size; ++i)
			if(!vis[i])
				dfs_helper(i, vis);
	}
}


//basic dfs recursive function.
void Graph::dfs_helper(int v, vector<bool>& vis){
	vis[v] = 1;
	SCC.back().push_back(v);

	for(auto u: adj[v])
		if(!vis[u])
			dfs_helper(u, vis);
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


//Strongly connected components
void Graph::strongly_connected_components(){
	vector<int> order_for_scc;
	compute_order_scc(order_for_scc);

	Graph rg = this->reverse_graph();
	rg.dfs(order_for_scc);
}


void Graph::compute_order_scc(vector<int>& order_for_scc){
	vector<bool> vis(size);

	for(int i = 1; i <= size; ++i)
		if(!vis[i])
			scc_helper(i, vis, order_for_scc);


	reverse(order_for_scc.begin(), order_for_scc.end());
}


//basic dfs recursive function.
void Graph::scc_helper(int v, vector<bool>& vis, vector<int>& order_for_scc){
	vis[v] = 1;

	for(auto u: adj[v])
		if(!vis[u])
			scc_helper(u, vis, order_for_scc);

	order_for_scc.push_back(v);
}


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