import heapq
from collections import deque

MAZE = [
    "S.....#.....",
    ".####.#.###.",
    ".#....#...#.",
    ".#.#######..",
    ".#.......#.#",
    ".#######.#..",
    ".......#.##.",
    "######.#....",
    "......#.###.",
    ".########...",
    "...........G",
]


def parse(maze):
    start = goal = None
    for r, row in enumerate(maze):
        for c, ch in enumerate(row):
            if ch == "S":
                start = (r, c)
            elif ch == "G":
                goal = (r, c)
    return start, goal


def neighbors(pos, maze):
    r, c = pos
    for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):  # up, right, down, left
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(maze) and 0 <= nc < len(maze[0]) and maze[nr][nc] != "#":
            yield (nr, nc)


def reconstruct(parent, goal):
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1]


def bfs(maze, start, goal):
    """Breadth-First Search: explores level by level; shortest path (unweighted)."""
    frontier = deque([start])
    parent = {start: None}
    expanded = 0
    while frontier:
        node = frontier.popleft()
        expanded += 1
        if node == goal:
            return reconstruct(parent, goal), expanded
        for nxt in neighbors(node, maze):
            if nxt not in parent:
                parent[nxt] = node
                frontier.append(nxt)
    return None, expanded


def dfs(maze, start, goal):
    """Depth-First Search: goes as deep as possible first; path is not guaranteed shortest."""
    frontier = [start]
    parent = {start: None}
    expanded = 0
    while frontier:
        node = frontier.pop()
        expanded += 1
        if node == goal:
            return reconstruct(parent, goal), expanded
        for nxt in neighbors(node, maze):
            if nxt not in parent:
                parent[nxt] = node
                frontier.append(nxt)
    return None, expanded


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(maze, start, goal):
    """A* Search: f(n) = g(n) + h(n) with Manhattan-distance heuristic (admissible)."""
    counter = 0  # tie-breaker so the heap never compares tuples of positions
    open_heap = [(manhattan(start, goal), counter, start)]
    parent = {start: None}
    g = {start: 0}
    closed = set()
    expanded = 0
    while open_heap:
        _, _, node = heapq.heappop(open_heap)
        if node in closed:
            continue
        closed.add(node)
        expanded += 1
        if node == goal:
            return reconstruct(parent, goal), expanded
        for nxt in neighbors(node, maze):
            new_g = g[node] + 1
            if nxt not in g or new_g < g[nxt]:
                g[nxt] = new_g
                parent[nxt] = node
                counter += 1
                heapq.heappush(open_heap, (new_g + manhattan(nxt, goal), counter, nxt))
    return None, expanded


def render(maze, path):
    on_path = set(path or [])
    lines = []
    for r, row in enumerate(maze):
        line = ""
        for c, ch in enumerate(row):
            line += "*" if (r, c) in on_path and ch == "." else ch
        lines.append(line)
    return "\n".join(lines)


def main():
    start, goal = parse(MAZE)
    print(f"Start: {start}  Goal: {goal}\n")
    results = []
    for name, algo in (("BFS", bfs), ("DFS", dfs), ("A*", astar)):
        path, expanded = algo(MAZE, start, goal)
        print(f"=== {name} ===")
        if path is None:
            print("No path found.\n")
            continue
        print(render(MAZE, path))
        print(f"Path length: {len(path) - 1} steps | Nodes expanded: {expanded}\n")
        results.append((name, len(path) - 1, expanded))

    print(f"{'Algorithm':<10}{'Path length':<14}{'Nodes expanded'}")
    for name, length, expanded in results:
        print(f"{name:<10}{length:<14}{expanded}")


if __name__ == "__main__":
    main()