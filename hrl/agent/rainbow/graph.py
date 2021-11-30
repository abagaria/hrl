from collections import defaultdict

class Graph:

    def __init__(self): 
        self.graph = defaultdict(list)
        self.fromnode = defaultdict(list)
        self.visited = defaultdict(list)
 
    def addEdge(self,u,v):
        self.graph[u].append(v)
 
    def BFS(self, s):
        visited = defaultdict(list)
        self.fromnode = defaultdict(list)

        queue = []
        queue.append(s)
        visited[s] = True
        while queue:
            s = queue.pop(0)
            for i in self.graph[s]:
                if i not in visited:
                    queue.append(i)
                    self.fromnode[i] = s
                    visited[i] = True

    def getAllNodes(self):
        return list(self.graph.keys())

    def getPath(self, start, goal):
        self.BFS(start)
        # pdb.set_trace()

        li = []
        curr = goal
        while curr != start:
            li.append(curr)
            curr = self.fromnode[curr]
        li.append(curr)
        li.reverse()
        return li