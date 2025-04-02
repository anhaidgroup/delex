def find_all_nodes(node):
    visited = {node}
    unvisited = [node]
    while len(unvisited):
        curr = unvisited.pop()
        for n in curr.iter_out():
            if n not in visited:
                visited.add(n)
                unvisited.append(n)

        for n in curr.iter_in():
            if n not in visited:
                visited.add(n)
                unvisited.append(n)

    return visited
    
def _reverse_dfs(node, visited):
    for tail in node.iter_in():
        if tail not in visited:
            _reverse_dfs(tail, visited)

    visited[node] = len(visited)

def topological_sort(sink_node):
    visited = {}
    _reverse_dfs(sink_node, visited)
    return list(visited)


def clone_graph(nodes):
    node_map = {n : n.clone_no_edges() for n in nodes}

    for old_node, new_node in node_map.items():
        for n in old_node.iter_out():
            if n in node_map:
                new_node.add_out_edge(node_map[n])

    return list(node_map.values())

def find_sink(node):
    if node.is_sink:
        return node
    
    queue = [node]
    visited = set()
    while len(queue):
        node = queue.pop()
        visited.add(node)
        for n in node.iter_out():
            if n.is_sink:
                return n
            elif n not in visited:
                queue.append(n)
