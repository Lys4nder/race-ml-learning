import heapq
import json
from typing import List, Tuple, Optional, Set, Dict

Grid = List[List[int]]  # 0 free, 1 wall, 2 start, 3 finish
Pos = Tuple[int, int]

MOVES = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar_simple(grid: Grid, start: Pos, goal: Pos) -> Optional[List[Pos]]:
    n = len(grid)
    open_heap = [(manhattan(start, goal), start)]
    came_from: Dict[Pos, Pos] = {}
    gscore = {start: 0}
    closed: Set[Pos] = set()

    if start == goal:
        return [start]

    while open_heap:
        current = heapq.heappop(open_heap)[1]

        if current == goal:
            path = [current]
            while path[-1] != start:
                path.append(came_from[path[-1]])
            path.reverse()
            return path

        closed.add(current)

        for dm in MOVES:
            nbr = (current[0] + dm[0], current[1] + dm[1])

            if not (0 <= nbr[0] < n and 0 <= nbr[1] < n) or grid[nbr[0]][nbr[1]] == 1:
                continue
            if nbr in closed:
                continue

            tentative = gscore[current] + 1
            prev = gscore.get(nbr)
            if prev is None or tentative < prev:
                came_from[nbr] = current
                gscore[nbr] = tentative
                f = tentative + manhattan(nbr, goal)
                heapq.heappush(open_heap, (f, nbr))

    return None


def compute_path_metrics(astar_path: Optional[List[Pos]], rl_path: Optional[List[Pos]], goal: Pos):
    def reached_goal(path: Optional[List[Pos]], goal: Pos):
        return bool(path) and path[-1] == goal

    astar_set = set(astar_path) if astar_path else set()
    rl_set = set(rl_path) if rl_path else set()

    tp = len(astar_set & rl_set)  # true positives: same cells visited
    fp = len(rl_set - astar_set)  # RL-only
    fn = len(astar_set - rl_set)  # A*-only
    total = len(astar_set | rl_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = tp / total if total > 0 else 0.0

    len_astar = len(astar_path) if astar_path else 0
    len_rl = len(rl_path) if rl_path else 0
    success_astar = reached_goal(astar_path, goal)
    success_rl = reached_goal(rl_path, goal)
    optimality_ratio = (len_rl / len_astar) if len_astar > 0 and len_rl > 0 else float("inf")

    return {
        "success_astar": success_astar,
        "success_rl": success_rl,
        "len_astar": len_astar,
        "len_rl": len_rl,
        "optimality_ratio": optimality_ratio,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy
    }


def best_rl_run(rl_runs: List[List[Pos]], goal: Pos) -> Optional[List[Pos]]:
    winning_runs = [run for run in rl_runs if run and run[-1] == goal]
    if not winning_runs:
        return None
    return min(winning_runs, key=len)


def load_map(path: str = "custom_map.json"):
    with open(path, "r") as f:
        data = json.load(f)
    grid = data["grid"]
    start = tuple(data["start_pos"])
    goal = tuple(data["finish_pos"])
    return grid, start, goal


def run_baseline_and_eval(custom_map_path="custom_map.json", rl_runs: Optional[List[List[Pos]]] = None):
    grid, start, goal = load_map(custom_map_path)
    astar_path = astar_simple(grid, start, goal)

    rl_path = best_rl_run(rl_runs, goal) if rl_runs else None

    if rl_path:
        with open("winning_run.json", "w") as f:
            json.dump(rl_path, f)

    metrics = compute_path_metrics(astar_path, rl_path, goal)

    print("\n=== Navigation Metrics ===")
    print(f"Reached goal (A*): {metrics['success_astar']}")
    print(f"Reached goal (RL): {metrics['success_rl']}")
    print(f"Path length (A*): {metrics['len_astar'] - 1}")
    print(f"Path length (RL): {metrics['len_rl'] - 1}")
    if metrics['optimality_ratio'] != float('inf'):
        print(f"Optimality ratio (RL/A*): {metrics['optimality_ratio']:.3f}")
    else:
        print("Optimality ratio: undefined (missing path)")
    print(f"Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | Accuracy: {metrics['accuracy']:.3f}")
    print("==========================")

    return astar_path, metrics


if __name__ == "__main__":
    try:
        try:
            with open("winning_run.json", "r") as f:
                rl_run = json.load(f)
                rl_runs = [[tuple(p) for p in rl_run]]
        except FileNotFoundError:
            rl_runs = None
            print("No winning_run.json found - skipping RL comparison.")

        astar_path, metrics = run_baseline_and_eval("custom_map.json", rl_runs=rl_runs)

    except FileNotFoundError:
        print("404: Map file not found.")
