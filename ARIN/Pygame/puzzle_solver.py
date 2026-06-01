"""8-Puzzle search — BFS, DFS, IDS, UCS, Greedy, A*, IDA*, leo đồi theo mã giả."""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass
from enum import Enum

GOAL: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 0)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
IDS_MAX_DEPTH = 80


class SearchAlgo(Enum):
    BFS = "BFS"
    DFS = "DFS"
    IDS = "IDS"
    UCS = "UCS"
    GREEDY = "Greedy"
    ASTAR = "A*"
    IDASTAR = "IDA*"
    SIMPLE_HILL = "Leo đồi đơn giản"
    STEEPEST_HILL = "Leo đồi dốc nhất"


class StepKind(Enum):
    INIT = "init"
    REMOVE = "remove"
    ADD = "add"
    EXPLORE = "explore"
    GOAL = "goal"
    FAIL = "fail"
    IDS_ROUND = "ids_round"
    CUTOFF = "cutoff"


@dataclass(frozen=True)
class SearchStep:
    kind: StepKind
    state: tuple[int, ...]
    reached: int
    frontier: int
    depth: int
    message: str
    depth_limit: int | None = None
    ids_round: int | None = None


@dataclass
class SearchResult:
    algo: SearchAlgo
    path: list[tuple[int, ...]]
    steps: list[SearchStep]
    nodes_expanded: int
    ids_rounds: int = 0
    final_depth_limit: int | None = None


def blank_index(state: tuple[int, ...]) -> int:
    return state.index(0)


def neighbors(state: tuple[int, ...]) -> list[tuple[int, ...]]:
    i = blank_index(state)
    r, c = divmod(i, 3)
    out: list[tuple[int, ...]] = []
    board = list(state)
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            j = nr * 3 + nc
            board[i], board[j] = board[j], board[i]
            out.append(tuple(board))
            board[i], board[j] = board[j], board[i]
    return out


def inversion_count(state: tuple[int, ...]) -> int:
    tiles = [x for x in state if x != 0]
    inv = 0
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            if tiles[i] > tiles[j]:
                inv += 1
    return inv


def is_solvable(state: tuple[int, ...]) -> bool:
    return inversion_count(state) % 2 == 0


def make_solvable(state: tuple[int, ...]) -> tuple[tuple[int, ...], bool]:
    if is_solvable(state):
        return state, False
    board = list(state)
    nz = [i for i, v in enumerate(board) if v != 0]
    board[nz[0]], board[nz[1]] = board[nz[1]], board[nz[0]]
    return tuple(board), True


def parse_custom_state(text: str) -> tuple[tuple[int, ...] | None, str]:
    raw = text.strip()
    if not raw:
        return None, "Chưa nhập gì."
    digits = [int(ch) for ch in raw if ch.isdigit()]
    if len(digits) != 9:
        return None, f"Cần đúng 9 chữ số (đang có {len(digits)})."
    if sorted(digits) != list(range(9)):
        return None, "Phải có đủ các số 0–8, mỗi số một lần."
    return tuple(digits), ""


def slide_move(prev: tuple[int, ...], new: tuple[int, ...]) -> tuple[int, int, int] | None:
    if sum(a != b for a, b in zip(prev, new)) != 2:
        return None
    bi_new = new.index(0)
    val = prev[bi_new]
    if val == 0:
        return None
    bi_prev = prev.index(0)
    if new[bi_prev] != val:
        return None
    return val, bi_new, bi_prev


def reconstruct(goal_node: tuple[int, ...], parent: dict[tuple[int, ...], tuple[int, ...] | None]) -> list[tuple[int, ...]]:
    path: list[tuple[int, ...]] = []
    cur: tuple[int, ...] | None = goal_node
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def manhattan(state: tuple[int, ...]) -> int:
    """Greedy.txt — h(n) = tổng khoảng cách Manhattan từng ô về vị trí đích."""
    total = 0
    for i, v in enumerate(state):
        if v == 0:
            continue
        gr, gc = divmod(v - 1, 3)
        r, c = divmod(i, 3)
        total += abs(r - gr) + abs(c - gc)
    return total


def swap_cost(parent: tuple[int, ...], child: tuple[int, ...]) -> int:
    """UCS.txt — chi phí bước = giá trị ô số được swap trực tiếp với ô trống (0)."""
    for i in range(9):
        if parent[i] != child[i] and parent[i] != 0:
            return parent[i]
    return 0


def _is_cycle(child: tuple[int, ...], node: tuple[int, ...], parent: dict[tuple[int, ...], tuple[int, ...] | None]) -> bool:
    cur: tuple[int, ...] | None = node
    while cur is not None:
        if cur == child:
            return True
        cur = parent[cur]
    return False


def _solve_bfs(start: tuple[int, ...], *, record_steps: bool) -> SearchResult | None:
    """BFS.txt: reached khi INSERT; frontier FIFO REMOVE."""
    if start == GOAL:
        step = SearchStep(StepKind.GOAL, start, 1, 0, 0, "Trạng thái ban đầu đã là đích.")
        return SearchResult(SearchAlgo.BFS, [start], [step] if record_steps else [], 0)

    frontier: deque[tuple[int, ...]] = deque([start])
    reached: set[tuple[int, ...]] = {start}
    parent: dict[tuple[int, ...], tuple[int, ...] | None] = {start: None}
    depth: dict[tuple[int, ...], int] = {start: 0}
    steps: list[SearchStep] = []
    nodes_expanded = 0

    def log(
        kind: StepKind,
        state: tuple[int, ...],
        msg: str,
        *,
        depth_limit: int | None = None,
        ids_round: int | None = None,
    ) -> None:
        if record_steps:
            steps.append(
                SearchStep(
                    kind,
                    state,
                    len(reached),
                    len(frontier),
                    depth.get(state, 0),
                    msg,
                    depth_limit=depth_limit,
                    ids_round=ids_round,
                )
            )

    log(StepKind.INIT, start, "Khởi tạo FIFO queue; reached ← {INITIAL}.")

    while frontier:
        node = frontier.popleft()
        nodes_expanded += 1
        log(StepKind.REMOVE, node, f"REMOVE từ frontier (FIFO), tầng {depth[node]}.")

        for child in neighbors(node):
            if child == GOAL:
                parent[child] = node
                path = reconstruct(child, parent)
                log(StepKind.GOAL, child, f"GOAL_TEST — tìm thấy đích sau {nodes_expanded} nút mở rộng.")
                return SearchResult(SearchAlgo.BFS, path, steps, nodes_expanded)

            if child not in reached:
                reached.add(child)
                parent[child] = node
                depth[child] = depth[node] + 1
                frontier.append(child)
                log(StepKind.ADD, child, f"INSERT vào frontier; reached ← reached + {{s}}, tầng {depth[child]}.")

        log(
            StepKind.EXPLORE,
            node,
            f"Hoàn tất mở rộng — |reached|={len(reached)}, |frontier|={len(frontier)}.",
        )

    log(StepKind.FAIL, start, "Frontier rỗng — failure.")
    return None


def _solve_dfs(start: tuple[int, ...], *, record_steps: bool) -> SearchResult | None:
    """DFS.txt: reached khi PUSH; frontier STACK POP."""
    if start == GOAL:
        step = SearchStep(StepKind.GOAL, start, 1, 0, 0, "Trạng thái ban đầu đã là đích.")
        return SearchResult(SearchAlgo.DFS, [start], [step] if record_steps else [], 0)

    frontier: list[tuple[int, ...]] = [start]
    reached: set[tuple[int, ...]] = {start}
    parent: dict[tuple[int, ...], tuple[int, ...] | None] = {start: None}
    depth: dict[tuple[int, ...], int] = {start: 0}
    steps: list[SearchStep] = []
    nodes_expanded = 0

    def log(kind: StepKind, state: tuple[int, ...], msg: str) -> None:
        if record_steps:
            steps.append(
                SearchStep(
                    kind,
                    state,
                    len(reached),
                    len(frontier),
                    depth.get(state, 0),
                    msg,
                )
            )

    log(StepKind.INIT, start, "Khởi tạo STACK; reached ← {INITIAL}.")

    while frontier:
        node = frontier.pop()
        nodes_expanded += 1
        log(StepKind.REMOVE, node, f"POP từ frontier (LIFO), tầng {depth[node]}.")

        for child in neighbors(node):
            if child == GOAL:
                parent[child] = node
                path = reconstruct(child, parent)
                log(StepKind.GOAL, child, f"GOAL_TEST — tìm thấy đích sau {nodes_expanded} nút mở rộng.")
                return SearchResult(SearchAlgo.DFS, path, steps, nodes_expanded)

            if child not in reached:
                reached.add(child)
                parent[child] = node
                depth[child] = depth[node] + 1
                frontier.append(child)
                log(StepKind.ADD, child, f"PUSH vào stack; reached ← reached + {{s}}, tầng {depth[child]}.")

        log(
            StepKind.EXPLORE,
            node,
            f"Hoàn tất mở rộng — |reached|={len(reached)}, |stack|={len(frontier)}.",
        )

    log(StepKind.FAIL, start, "Stack rỗng — failure.")
    return None


def _depth_limited_search(
    start: tuple[int, ...],
    limit: int,
    ids_round: int,
    *,
    record_steps: bool,
    steps: list[SearchStep],
) -> tuple[SearchResult | None, bool]:
    """DEPTH-LIMITED-SEARCH (IDS.txt). Trả về (kết quả, có_cutoff)."""
    frontier: list[tuple[int, ...]] = [start]
    parent: dict[tuple[int, ...], tuple[int, ...] | None] = {start: None}
    depth_map: dict[tuple[int, ...], int] = {start: 0}
    nodes_expanded = 0
    cutoff = False

    def log(kind: StepKind, state: tuple[int, ...], msg: str) -> None:
        if record_steps:
            steps.append(
                SearchStep(
                    kind,
                    state,
                    0,
                    len(frontier),
                    depth_map.get(state, 0),
                    msg,
                    depth_limit=limit,
                    ids_round=ids_round,
                )
            )

    while frontier:
        node = frontier.pop()
        nodes_expanded += 1
        log(StepKind.REMOVE, node, f"POP (DLS, limit={limit}), độ sâu {depth_map[node]}.")

        if node == GOAL:
            path = reconstruct(node, parent)
            log(StepKind.GOAL, node, f"IS-GOAL tại độ sâu {depth_map[node]}.")
            return (
                SearchResult(
                    SearchAlgo.IDS,
                    path,
                    steps,
                    nodes_expanded,
                    ids_rounds=ids_round,
                    final_depth_limit=limit,
                ),
                False,
            )

        if depth_map[node] >= limit:
            cutoff = True
            log(StepKind.CUTOFF, node, f"DEPTH(node) ≥ {limit} — gán cutoff.")
            continue

        for child in neighbors(node):
            if _is_cycle(child, node, parent):
                continue
            parent[child] = node
            depth_map[child] = depth_map[node] + 1
            frontier.append(child)
            log(StepKind.ADD, child, f"PUSH con (không cycle), độ sâu {depth_map[child]}.")

    return None, cutoff


def _solve_ids(start: tuple[int, ...], *, record_steps: bool) -> SearchResult | None:
    """ITERATIVE-DEEPENING-SEARCH (IDS.txt)."""
    if start == GOAL:
        step = SearchStep(StepKind.GOAL, start, 1, 0, 0, "Trạng thái ban đầu đã là đích.")
        return SearchResult(SearchAlgo.IDS, [start], [step] if record_steps else [], 0)

    steps: list[SearchStep] = []
    total_expanded = 0

    for depth_limit in range(IDS_MAX_DEPTH + 1):
        ids_round = depth_limit + 1
        if record_steps:
            steps.append(
                SearchStep(
                    StepKind.IDS_ROUND,
                    start,
                    0,
                    1,
                    0,
                    f"Vòng IDS: depth = {depth_limit} → gọi DEPTH-LIMITED-SEARCH.",
                    depth_limit=depth_limit,
                    ids_round=ids_round,
                )
            )

        round_steps: list[SearchStep] = []
        result, cutoff = _depth_limited_search(
            start,
            depth_limit,
            ids_round,
            record_steps=record_steps,
            steps=round_steps,
        )
        total_expanded += sum(1 for s in round_steps if s.kind is StepKind.REMOVE)

        if result is not None:
            if record_steps:
                steps.extend(round_steps)
            result.nodes_expanded = total_expanded
            result.steps = steps
            result.ids_rounds = ids_round
            result.final_depth_limit = depth_limit
            return result

        if record_steps:
            steps.extend(round_steps)
            steps.append(
                SearchStep(
                    StepKind.CUTOFF,
                    start,
                    0,
                    0,
                    0,
                    f"DLS trả về cutoff — tăng depth lên {depth_limit + 1}.",
                    depth_limit=depth_limit,
                    ids_round=ids_round,
                )
            )

        if not cutoff:
            break

    if record_steps:
        steps.append(SearchStep(StepKind.FAIL, start, 0, 0, 0, "IDS failure — vượt giới hạn độ sâu."))
    return None


def _solve_ucs(start: tuple[int, ...], *, record_steps: bool) -> SearchResult | None:
    """UCS.txt: frontier = priority queue theo g(n); chi phí bước = ô swap với 0."""
    if start == GOAL:
        step = SearchStep(StepKind.GOAL, start, 1, 0, 0, "Trạng thái ban đầu đã là đích.")
        return SearchResult(SearchAlgo.UCS, [start], [step] if record_steps else [], 0)

    counter = 0
    heap: list[tuple[int, int, tuple[int, ...]]] = []
    heapq.heappush(heap, (0, counter, start))
    counter += 1

    reached: dict[tuple[int, ...], int] = {start: 0}
    parent: dict[tuple[int, ...], tuple[int, ...] | None] = {start: None}
    depth: dict[tuple[int, ...], int] = {start: 0}
    steps: list[SearchStep] = []
    nodes_expanded = 0

    def log(kind: StepKind, state: tuple[int, ...], msg: str) -> None:
        if record_steps:
            steps.append(
                SearchStep(
                    kind,
                    state,
                    len(reached),
                    len(heap),
                    depth.get(state, 0),
                    msg,
                )
            )

    log(StepKind.INIT, start, "Khởi tạo priority queue theo g(n)=0; reached[INITIAL]←0.")

    while heap:
        g, _, node = heapq.heappop(heap)
        if g > reached.get(node, 10**9):
            continue

        nodes_expanded += 1
        log(StepKind.REMOVE, node, f"REMOVE-MIN g(n)={g}, tầng {depth[node]}.")

        if node == GOAL:
            path = reconstruct(node, parent)
            log(StepKind.GOAL, node, f"GOAL — g(n)={g}, sau {nodes_expanded} nút mở rộng.")
            return SearchResult(SearchAlgo.UCS, path, steps, nodes_expanded)

        for child in neighbors(node):
            step = swap_cost(node, child)
            new_g = g + step
            if child not in reached or new_g < reached[child]:
                reached[child] = new_g
                parent[child] = node
                depth[child] = depth[node] + 1
                heapq.heappush(heap, (new_g, counter, child))
                counter += 1
                log(
                    StepKind.ADD,
                    child,
                    f"INSERT g={new_g} (chi phí bước +{step}, ô số {step}).",
                )

        log(
            StepKind.EXPLORE,
            node,
            f"Hoàn tất mở rộng — |reached|={len(reached)}, |PQ|={len(heap)}.",
        )

    log(StepKind.FAIL, start, "Priority queue rỗng — failure.")
    return None


def _solve_greedy(start: tuple[int, ...], *, record_steps: bool) -> SearchResult | None:
    """Greedy.txt: frontier = priority queue theo h(n) Manhattan; reached khi INSERT."""
    if start == GOAL:
        step = SearchStep(StepKind.GOAL, start, 1, 0, 0, "Trạng thái ban đầu đã là đích.")
        return SearchResult(SearchAlgo.GREEDY, [start], [step] if record_steps else [], 0)

    counter = 0
    h0 = manhattan(start)
    heap: list[tuple[int, int, tuple[int, ...]]] = []
    heapq.heappush(heap, (h0, counter, start))
    counter += 1

    reached: set[tuple[int, ...]] = {start}
    expanded: set[tuple[int, ...]] = set()
    parent: dict[tuple[int, ...], tuple[int, ...] | None] = {start: None}
    depth: dict[tuple[int, ...], int] = {start: 0}
    steps: list[SearchStep] = []
    nodes_expanded = 0

    def log(kind: StepKind, state: tuple[int, ...], msg: str) -> None:
        if record_steps:
            steps.append(
                SearchStep(
                    kind,
                    state,
                    len(reached),
                    len(heap),
                    depth.get(state, 0),
                    msg,
                )
            )

    log(StepKind.INIT, start, f"Khởi tạo PQ theo h(n); h(INITIAL)={h0}.")

    while heap:
        h, _, node = heapq.heappop(heap)
        if node in expanded:
            continue
        expanded.add(node)

        nodes_expanded += 1
        log(StepKind.REMOVE, node, f"REMOVE-MIN h(n)={h}, tầng {depth[node]}.")

        if node == GOAL:
            path = reconstruct(node, parent)
            log(StepKind.GOAL, node, f"GOAL — h(n)={h}, sau {nodes_expanded} nút mở rộng.")
            return SearchResult(SearchAlgo.GREEDY, path, steps, nodes_expanded)

        for child in neighbors(node):
            if child not in reached:
                reached.add(child)
                parent[child] = node
                depth[child] = depth[node] + 1
                ch = manhattan(child)
                heapq.heappush(heap, (ch, counter, child))
                counter += 1
                log(StepKind.ADD, child, f"INSERT h={ch}; reached ← reached + {{s}}.")

        log(
            StepKind.EXPLORE,
            node,
            f"Hoàn tất mở rộng — |reached|={len(reached)}, |PQ|={len(heap)}.",
        )

    log(StepKind.FAIL, start, "Priority queue rỗng — failure (Greedy có thể không tìm được đích).")
    return None


def _solve_astar(start: tuple[int, ...], *, record_steps: bool) -> SearchResult | None:
    """A_star.txt: frontier = priority queue theo f(n)=g(n)+h(n), h là Manhattan."""
    if start == GOAL:
        step = SearchStep(StepKind.GOAL, start, 1, 0, 0, "Trạng thái ban đầu đã là đích.")
        return SearchResult(SearchAlgo.ASTAR, [start], [step] if record_steps else [], 0)

    counter = 0
    g0 = 0
    h0 = manhattan(start)
    heap: list[tuple[int, int, int, tuple[int, ...]]] = []
    heapq.heappush(heap, (g0 + h0, g0, counter, start))
    counter += 1

    reached: dict[tuple[int, ...], int] = {start: g0}
    expanded: set[tuple[int, ...]] = set()
    parent: dict[tuple[int, ...], tuple[int, ...] | None] = {start: None}
    depth: dict[tuple[int, ...], int] = {start: 0}
    steps: list[SearchStep] = []
    nodes_expanded = 0

    def log(kind: StepKind, state: tuple[int, ...], msg: str) -> None:
        if record_steps:
            steps.append(
                SearchStep(
                    kind,
                    state,
                    len(reached),
                    len(heap),
                    depth.get(state, 0),
                    msg,
                )
            )

    log(StepKind.INIT, start, f"Khởi tạo PQ theo f=g+h; g=0, h={h0}, f={h0}.")

    while heap:
        f, g, _, node = heapq.heappop(heap)
        if node in expanded:
            continue
        if g > reached.get(node, 10**9):
            continue
        expanded.add(node)

        nodes_expanded += 1
        log(StepKind.REMOVE, node, f"REMOVE-MIN f={f} (g={g}, h={f-g}), tầng {depth[node]}.")

        if node == GOAL:
            path = reconstruct(node, parent)
            log(StepKind.GOAL, node, f"GOAL — f={f}, g={g}, sau {nodes_expanded} nút mở rộng.")
            return SearchResult(SearchAlgo.ASTAR, path, steps, nodes_expanded)

        for child in neighbors(node):
            new_g = g + 1
            if child not in reached or new_g < reached[child]:
                reached[child] = new_g
                parent[child] = node
                depth[child] = depth[node] + 1
                ch = manhattan(child)
                heapq.heappush(heap, (new_g + ch, new_g, counter, child))
                counter += 1
                log(StepKind.ADD, child, f"INSERT f={new_g + ch} (g={new_g}, h={ch}).")

        log(
            StepKind.EXPLORE,
            node,
            f"Hoàn tất mở rộng — |reached|={len(reached)}, |PQ|={len(heap)}.",
        )

    log(StepKind.FAIL, start, "Priority queue rỗng — failure.")
    return None


def _ida_star_dfs(
    node: tuple[int, ...],
    g: int,
    threshold: int,
    path: list[tuple[int, ...]],
    path_set: set[tuple[int, ...]],
    parent: dict[tuple[int, ...], tuple[int, ...] | None],
    steps: list[SearchStep],
    *,
    record_steps: bool,
    stats: dict[str, int],
    round_no: int,
) -> tuple[int, tuple[int, ...] | None]:
    h = manhattan(node)
    f = g + h

    if record_steps:
        steps.append(
            SearchStep(
                StepKind.REMOVE,
                node,
                len(path_set),
                len(path),
                g,
                f"POP node với f={f} (g={g}, h={h}), threshold={threshold}.",
                depth_limit=threshold,
                ids_round=round_no,
            )
        )

    if f > threshold:
        if record_steps:
            steps.append(
                SearchStep(
                    StepKind.CUTOFF,
                    node,
                    len(path_set),
                    len(path),
                    g,
                    f"f={f} > threshold={threshold}; cập nhật minimum.",
                    depth_limit=threshold,
                    ids_round=round_no,
                )
            )
        return f, None

    if node == GOAL:
        if record_steps:
            steps.append(
                SearchStep(
                    StepKind.GOAL,
                    node,
                    len(path_set),
                    len(path),
                    g,
                    f"IS-GOAL tại g={g}, f={f}.",
                    depth_limit=threshold,
                    ids_round=round_no,
                )
            )
        return -1, node

    stats["expanded"] += 1
    minimum = 10**9

    for child in neighbors(node):
        if child in path_set:
            continue
        parent[child] = node
        path.append(child)
        path_set.add(child)
        if record_steps:
            cg = g + 1
            ch = manhattan(child)
            steps.append(
                SearchStep(
                    StepKind.ADD,
                    child,
                    len(path_set),
                    len(path),
                    cg,
                    f"PUSH child f={cg + ch} (g={cg}, h={ch}).",
                    depth_limit=threshold,
                    ids_round=round_no,
                )
            )
        t, goal = _ida_star_dfs(
            child,
            g + 1,
            threshold,
            path,
            path_set,
            parent,
            steps,
            record_steps=record_steps,
            stats=stats,
            round_no=round_no,
        )
        if goal is not None:
            return -1, goal
        if t < minimum:
            minimum = t
        path_set.remove(child)
        path.pop()

    if record_steps:
        steps.append(
            SearchStep(
                StepKind.EXPLORE,
                node,
                len(path_set),
                len(path),
                g,
                "Hoàn tất mở rộng node trong contour hiện tại.",
                depth_limit=threshold,
                ids_round=round_no,
            )
        )
    return minimum, None


def _solve_idastar(start: tuple[int, ...], *, record_steps: bool) -> SearchResult | None:
    """IDA_star.txt: lặp theo ngưỡng f, DFS theo contour với f(n)=g(n)+h(n)."""
    if start == GOAL:
        step = SearchStep(StepKind.GOAL, start, 1, 0, 0, "Trạng thái ban đầu đã là đích.")
        return SearchResult(SearchAlgo.IDASTAR, [start], [step] if record_steps else [], 0)

    threshold = manhattan(start)
    round_no = 0
    steps: list[SearchStep] = []
    nodes_expanded = 0

    while True:
        round_no += 1
        parent: dict[tuple[int, ...], tuple[int, ...] | None] = {start: None}
        path = [start]
        path_set = {start}
        stats = {"expanded": 0}

        if record_steps:
            steps.append(
                SearchStep(
                    StepKind.IDS_ROUND,
                    start,
                    1,
                    1,
                    0,
                    f"IDA* vòng {round_no}: threshold={threshold}.",
                    depth_limit=threshold,
                    ids_round=round_no,
                )
            )

        t, goal = _ida_star_dfs(
            start,
            0,
            threshold,
            path,
            path_set,
            parent,
            steps,
            record_steps=record_steps,
            stats=stats,
            round_no=round_no,
        )
        nodes_expanded += stats["expanded"]

        if goal is not None:
            path_to_goal = reconstruct(goal, parent)
            return SearchResult(
                SearchAlgo.IDASTAR,
                path_to_goal,
                steps,
                nodes_expanded,
                ids_rounds=round_no,
                final_depth_limit=threshold,
            )

        if t >= 10**9:
            if record_steps:
                steps.append(
                    SearchStep(
                        StepKind.FAIL,
                        start,
                        0,
                        0,
                        0,
                        "IDA* failure — không còn ngưỡng kế tiếp.",
                        depth_limit=threshold,
                        ids_round=round_no,
                    )
                )
            return None

        threshold = t


def _hill_value(state: tuple[int, ...]) -> int:
    """Chi phí = Manhattan; leo đồi tối ưu hóa bằng cách giảm h (VALUE tốt hơn khi h nhỏ hơn)."""
    return manhattan(state)


def _solve_simple_hill(start: tuple[int, ...], *, record_steps: bool) -> SearchResult | None:
    """Simple-Hill-Climbing.txt: chọn láng giềng đầu tiên có h nhỏ hơn h(current)."""
    if start == GOAL:
        step = SearchStep(StepKind.GOAL, start, 1, 1, 0, "Trạng thái ban đầu đã là đích.")
        return SearchResult(SearchAlgo.SIMPLE_HILL, [start], [step] if record_steps else [], 0)

    current = start
    parent: dict[tuple[int, ...], tuple[int, ...] | None] = {start: None}
    depth: dict[tuple[int, ...], int] = {start: 0}
    steps: list[SearchStep] = []
    nodes_expanded = 0

    def log(kind: StepKind, state: tuple[int, ...], msg: str) -> None:
        if record_steps:
            steps.append(
                SearchStep(
                    kind,
                    state,
                    len(depth),
                    1,
                    depth.get(state, depth.get(current, 0)),
                    msg,
                )
            )

    h0 = _hill_value(start)
    log(StepKind.INIT, start, f"current ← INITIAL; h(n)={h0} (Manhattan).")

    while True:
        if current == GOAL:
            path = reconstruct(current, parent)
            log(StepKind.GOAL, current, f"Đạt đích sau {nodes_expanded} bước leo đồi.")
            return SearchResult(SearchAlgo.SIMPLE_HILL, path, steps, nodes_expanded)

        nodes_expanded += 1
        h_cur = _hill_value(current)
        child_list = neighbors(current)
        scan = ", ".join(f"h={_hill_value(c)}" for c in child_list) or "(không có)"
        log(
            StepKind.EXPLORE,
            current,
            f"Duyệt EXPAND(current): [{scan}] — ưu tiên h nhỏ hơn {h_cur}.",
        )

        found_better = False
        for child in child_list:
            h_ch = _hill_value(child)
            if h_ch < h_cur:
                parent[child] = current
                depth[child] = depth[current] + 1
                log(
                    StepKind.ADD,
                    child,
                    f"Láng giềng đầu tiên tốt hơn: h {h_cur}→{h_ch}; current ← neighbor; break.",
                )
                current = child
                found_better = True
                break

        if not found_better:
            path = reconstruct(current, parent)
            log(
                StepKind.FAIL,
                current,
                f"Không còn láng giềng tốt hơn — cực tiểu cục bộ h={h_cur}.",
            )
            return SearchResult(SearchAlgo.SIMPLE_HILL, path, steps, nodes_expanded)


def _solve_steepest_hill(start: tuple[int, ...], *, record_steps: bool) -> SearchResult | None:
    """Steepest-ascent-hill-climbing.txt: chọn láng giềng có h nhỏ nhất trong một vòng lặp."""
    if start == GOAL:
        step = SearchStep(StepKind.GOAL, start, 1, 1, 0, "Trạng thái ban đầu đã là đích.")
        return SearchResult(SearchAlgo.STEEPEST_HILL, [start], [step] if record_steps else [], 0)

    current = start
    parent: dict[tuple[int, ...], tuple[int, ...] | None] = {start: None}
    depth: dict[tuple[int, ...], int] = {start: 0}
    steps: list[SearchStep] = []
    nodes_expanded = 0

    def log(kind: StepKind, state: tuple[int, ...], msg: str) -> None:
        if record_steps:
            steps.append(
                SearchStep(
                    kind,
                    state,
                    len(depth),
                    1,
                    depth.get(state, depth.get(current, 0)),
                    msg,
                )
            )

    h0 = _hill_value(start)
    log(StepKind.INIT, start, f"current ← INITIAL; h(n)={h0} (Manhattan).")

    while True:
        if current == GOAL:
            path = reconstruct(current, parent)
            log(StepKind.GOAL, current, f"Đạt đích sau {nodes_expanded} bước leo đồi.")
            return SearchResult(SearchAlgo.STEEPEST_HILL, path, steps, nodes_expanded)

        nodes_expanded += 1
        h_cur = _hill_value(current)
        child_list = neighbors(current)
        best = current
        best_h = h_cur

        for child in child_list:
            h_ch = _hill_value(child)
            if h_ch < best_h:
                best = child
                best_h = h_ch

        scan = ", ".join(f"h={_hill_value(c)}" for c in child_list) or "(không có)"
        log(
            StepKind.EXPLORE,
            current,
            f"Duyệt EXPAND(current): [{scan}] — chọn h nhỏ nhất (best_h={best_h}).",
        )

        if best_h >= h_cur:
            path = reconstruct(current, parent)
            log(
                StepKind.FAIL,
                current,
                f"Không cải thiện được — cực tiểu cục bộ h={h_cur}.",
            )
            return SearchResult(SearchAlgo.STEEPEST_HILL, path, steps, nodes_expanded)

        parent[best] = current
        depth[best] = depth[current] + 1
        log(
            StepKind.ADD,
            best,
            f"Láng giềng tốt nhất: h {h_cur}→{best_h}; current ← best.",
        )
        current = best


def solve(start: tuple[int, ...], algo: SearchAlgo, *, record_steps: bool = True) -> SearchResult | None:
    if not is_solvable(start):
        return None
    if algo is SearchAlgo.BFS:
        return _solve_bfs(start, record_steps=record_steps)
    if algo is SearchAlgo.DFS:
        return _solve_dfs(start, record_steps=record_steps)
    if algo is SearchAlgo.IDS:
        return _solve_ids(start, record_steps=record_steps)
    if algo is SearchAlgo.UCS:
        return _solve_ucs(start, record_steps=record_steps)
    if algo is SearchAlgo.GREEDY:
        return _solve_greedy(start, record_steps=record_steps)
    if algo is SearchAlgo.ASTAR:
        return _solve_astar(start, record_steps=record_steps)
    if algo is SearchAlgo.IDASTAR:
        return _solve_idastar(start, record_steps=record_steps)
    if algo is SearchAlgo.SIMPLE_HILL:
        return _solve_simple_hill(start, record_steps=record_steps)
    return _solve_steepest_hill(start, record_steps=record_steps)


def solve_bfs(start: tuple[int, ...], *, record_steps: bool = True) -> SearchResult | None:
    """Alias tương thích — chỉ gọi BFS (theo BFS.txt)."""
    return solve(start, SearchAlgo.BFS, record_steps=record_steps)
