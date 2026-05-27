"""
8-Puzzle — Pygame UI trực quan hóa BFS / DFS / IDS / UCS / Greedy / A* / IDA*.
Run: python puzzle8_game.py
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from enum import Enum, auto

import pygame

from puzzle_solver import (
    GOAL,
    SearchAlgo,
    SearchResult,
    SearchStep,
    StepKind,
    blank_index,
    inversion_count,
    is_solvable,
    make_solvable,
    neighbors,
    parse_custom_state,
    slide_move,
    solve,
)

SLIDE_DURATION = 0.75
SNAP_STEP_DELAY = 0.04
LOG_HISTORY_MAX = 80
LOG_VISIBLE_LINES = 7

# --- Kích thước & vùng bố cục ---
MARGIN = 20
HEADER_H = 56
SECTION_GAP = 10
SECTION_PAD = 12
BTN_ROW_H = 28
BTN_GAP = 6
CTRL_ROWS = 4
CTRL_BOTTOM_PAD = 10

LEFT_W = 420
BOARD_SIZE = 360
BOARD_X = MARGIN + (LEFT_W - BOARD_SIZE) // 2
BOARD_Y = HEADER_H + 12

RIGHT_X = MARGIN + LEFT_W + 16

BAR_SECTION_H = 128
ALGO_SECTION_H = 72
LOG_SECTION_H = 118
INFO_MAX_LINES = 8
LEGEND_ROW_H = 22
LEGEND_ICON = 14
LEGEND_TEXT_GAP = 8

COLORS = {
    "bg": (18, 22, 35),
    "section": (28, 34, 52),
    "section_border": (55, 65, 95),
    "title": (230, 236, 255),
    "text": (180, 190, 220),
    "muted": (130, 140, 170),
    "accent": (99, 179, 237),
    "accent_dim": (60, 110, 160),
    "success": (72, 199, 142),
    "tile": (45, 55, 80),
    "tile_hover": (58, 72, 105),
    "tile_blank": (32, 38, 58),
    "tile_num": (240, 245, 255),
    "tile_goal": (56, 90, 70),
    "highlight_dequeue": (255, 200, 87),
    "highlight_enqueue": (120, 220, 180),
    "btn": (52, 64, 96),
    "btn_hover": (72, 88, 128),
    "btn_active": (99, 179, 237),
    "btn_text": (240, 245, 255),
    "bar_explored": (99, 179, 237),
    "bar_frontier": (255, 183, 77),
    "overlay": (10, 14, 28, 210),
    "input_bg": (22, 28, 44),
    "input_border": (99, 179, 237),
    "input_error": (255, 107, 107),
    "log_active": (255, 214, 102),
    "log_active_bg": (55, 48, 28),
}


@dataclass
class SlideAnim:
    value: int
    from_idx: int
    to_idx: int
    t: float = 0.0


class Mode(Enum):
    PLAY = auto()
    VISUALIZE = auto()
    SOLUTION = auto()


class Button:
    def __init__(self, rect: pygame.Rect, label: str, *, toggle: bool = False):
        self.rect = rect
        self.label = label
        self.toggle = toggle
        self.selected = False
        self.enabled = True

    def draw(self, surf: pygame.Surface, font: pygame.font.Font) -> None:
        if not self.enabled:
            bg, fg = (40, 45, 60), (100, 105, 120)
        elif self.toggle and self.selected:
            bg, fg = COLORS["btn_active"], (20, 25, 40)
        elif self.rect.collidepoint(pygame.mouse.get_pos()):
            bg, fg = COLORS["btn_hover"], COLORS["btn_text"]
        else:
            bg, fg = COLORS["btn"], COLORS["btn_text"]
        pygame.draw.rect(surf, bg, self.rect, border_radius=8)
        pygame.draw.rect(surf, COLORS["section_border"], self.rect, 2, border_radius=8)
        txt = font.render(self.label, True, fg)
        surf.blit(txt, txt.get_rect(center=self.rect.center))

    def hit(self, pos: tuple[int, int]) -> bool:
        return self.enabled and self.rect.collidepoint(pos)


class TextFlow:
    """Bố cục dọc trong một vùng — tránh chồng chữ."""

    def __init__(self, x: int, y: int, width: int, *, line_gap: int = 6):
        self.x = x
        self.y = y
        self.width = width
        self.line_gap = line_gap

    def skip(self, px: int) -> None:
        self.y += px

    def heading(self, surf: pygame.Surface, font: pygame.font.Font, text: str) -> None:
        t = font.render(text, True, COLORS["title"])
        surf.blit(t, (self.x, self.y))
        self.y += t.get_height() + 10

    def line(self, surf: pygame.Surface, font: pygame.font.Font, text: str, color=None) -> None:
        color = color or COLORS["text"]
        t = font.render(text, True, color)
        surf.blit(t, (self.x, self.y))
        self.y += t.get_height() + self.line_gap

    def wrap(self, surf: pygame.Surface, font: pygame.font.Font, text: str, color=None) -> None:
        color = color or COLORS["text"]
        for row in wrap_words(font, text, self.width):
            t = font.render(row, True, color)
            surf.blit(t, (self.x, self.y))
            self.y += t.get_height() + self.line_gap

    @property
    def cursor(self) -> int:
        return self.y


def wrap_words(font: pygame.font.Font, text: str, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        test = " ".join(cur + [w])
        if font.size(test)[0] <= max_width:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines


def draw_section_box(surf: pygame.Surface, rect: pygame.Rect, title: str, font: pygame.font.Font) -> int:
    """Vẽ khung section; trả về y bắt đầu nội dung bên trong."""
    pygame.draw.rect(surf, COLORS["section"], rect, border_radius=12)
    pygame.draw.rect(surf, COLORS["section_border"], rect, 2, border_radius=12)
    t = font.render(title, True, COLORS["title"])
    surf.blit(t, (rect.x + SECTION_PAD, rect.y + SECTION_PAD))
    return rect.y + SECTION_PAD + t.get_height() + 8


def draw_legend_row(
    surf: pygame.Surface,
    x: int,
    y: int,
    color: tuple[int, int, int],
    label: str,
    font: pygame.font.Font,
) -> int:
    """Icon bên trái, chữ bên phải — không chồng lên nhau."""
    pygame.draw.rect(
        surf, color, (x, y + 4, LEGEND_ICON, LEGEND_ICON), border_radius=3
    )
    text_x = x + LEGEND_ICON + LEGEND_TEXT_GAP
    t = font.render(label, True, COLORS["text"])
    text_y = y + 4 + (LEGEND_ICON - t.get_height()) // 2
    surf.blit(t, (text_x, text_y))
    return y + LEGEND_ROW_H


def draw_metric_bar(
    surf: pygame.Surface,
    x: int,
    y: int,
    width: int,
    label: str,
    value: int,
    color: tuple[int, int, int],
    max_value: int,
    font: pygame.font.Font,
) -> int:
    label_surf = font.render(label, True, COLORS["text"])
    surf.blit(label_surf, (x, y))
    val_surf = font.render(str(value), True, COLORS["accent"])
    surf.blit(val_surf, (x + width - val_surf.get_width(), y))
    bar_y = y + label_surf.get_height() + 6
    fill = int(width * min(value / max(max_value, 1), 1.0))
    pygame.draw.rect(surf, (32, 38, 54), (x, bar_y, width, 12), border_radius=6)
    if fill:
        pygame.draw.rect(surf, color, (x, bar_y, max(fill, 4), 12), border_radius=6)
    return bar_y + 12 + 10


def section_title_height(font: pygame.font.Font) -> int:
    return SECTION_PAD + font.get_height() + 6


def calc_controls_bottom() -> int:
    """Y ngay dưới khối nút điều khiển."""
    rows_h = CTRL_ROWS * BTN_ROW_H + (CTRL_ROWS - 1) * BTN_GAP
    return HEADER_H + 8 + rows_h + CTRL_BOTTOM_PAD


def calc_info_height(font: pygame.font.Font, line_count: int, *, two_cols: bool = True) -> int:
    line_h = font.get_height() + 4
    rows = (line_count + 1) // 2 if two_cols else line_count
    return section_title_height(font) + rows * line_h + SECTION_PAD + 4


def fit_window_size(content_top: int, info_h: int) -> tuple[int, int]:
    legend_h = section_title_height(pygame.font.SysFont("segoeui", 18, bold=True)) + 2 * LEGEND_ROW_H + SECTION_PAD
    left_need = BOARD_Y + BOARD_SIZE + 16 + legend_h + MARGIN
    right_need = (
        content_top
        + info_h
        + SECTION_GAP
        + BAR_SECTION_H
        + SECTION_GAP
        + ALGO_SECTION_H
        + SECTION_GAP
        + LOG_SECTION_H
        + MARGIN
    )
    need_h = max(left_need, right_need, 680)
    need_w = 1120
    info = pygame.display.Info()
    max_w = max(900, int(info.current_w * 0.96))
    max_h = max(600, int(info.current_h * 0.94))
    return min(need_w, max_w), min(need_h, max_h)


def format_search_log(step: SearchStep, algo: SearchAlgo) -> str:
    tag = algo.value
    d = step.depth
    if algo in (SearchAlgo.UCS, SearchAlgo.GREEDY, SearchAlgo.ASTAR, SearchAlgo.IDASTAR) and step.message:
        if step.kind in (
            StepKind.INIT,
            StepKind.REMOVE,
            StepKind.ADD,
            StepKind.EXPLORE,
            StepKind.GOAL,
            StepKind.FAIL,
        ):
            return f"[{tag}] {step.message}"
    if step.kind is StepKind.INIT:
        return f"[{tag}] Khởi tạo — |frontier|={step.frontier}."
    if step.kind is StepKind.REMOVE:
        verb = "REMOVE" if algo is SearchAlgo.BFS else "POP"
        return f"[{tag}] {verb} nút tầng {d} (|reached|={step.reached}, |frontier|={step.frontier})"
    if step.kind is StepKind.ADD:
        verb = "INSERT" if algo is SearchAlgo.BFS else "PUSH"
        return f"[{tag}] {verb} con tầng {d} (|frontier|={step.frontier})"
    if step.kind is StepKind.EXPLORE:
        return f"[{tag}] Mở rộng xong tầng {d} — |reached|={step.reached}, |frontier|={step.frontier}"
    if step.kind is StepKind.IDS_ROUND:
        return f"[{tag}] Vòng {step.ids_round}: depth_limit={step.depth_limit}"
    if step.kind is StepKind.CUTOFF:
        lim = step.depth_limit if step.depth_limit is not None else "?"
        return f"[{tag}] Cutoff tại limit={lim} (độ sâu {d})"
    if step.kind is StepKind.GOAL:
        return f"[{tag}] Đã tìm thấy đích tại tầng {d}!"
    if step.kind is StepKind.FAIL:
        return f"[{tag}] Không tìm thấy lời giải."
    return f"[{tag}] {step.message}"


def draw_text_columns(
    surf: pygame.Surface,
    x: int,
    y: int,
    col_w: int,
    lines: list[str],
    font: pygame.font.Font,
    *,
    line_gap: int = 3,
) -> int:
    """Vẽ danh sách dòng chia 2 cột; trả về y cuối."""
    if not lines:
        return y
    half = (len(lines) + 1) // 2
    left, right = lines[:half], lines[half:]
    line_h = font.get_height() + line_gap
    rows = max(len(left), len(right))
    for i in range(rows):
        if i < len(left):
            surf.blit(font.render(left[i], True, COLORS["text"]), (x, y + i * line_h))
        if i < len(right):
            surf.blit(font.render(right[i], True, COLORS["text"]), (x + col_w, y + i * line_h))
    return y + rows * line_h


class PuzzleApp:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("8-Puzzle · BFS / DFS / IDS / UCS / Greedy / A* / IDA*")

        self.font_title = pygame.font.SysFont("segoeui", 26, bold=True)
        self.font_heading = pygame.font.SysFont("segoeui", 16, bold=True)
        self.font_btn = pygame.font.SysFont("segoeui", 14)
        self.font_body = pygame.font.SysFont("segoeui", 13)
        self.font_sm = pygame.font.SysFont("consolas", 13)
        self.font_tile = pygame.font.SysFont("segoeui", 42, bold=True)

        self.content_top = calc_controls_bottom()
        info_h = calc_info_height(self.font_sm, INFO_MAX_LINES)
        self.W, self.H = fit_window_size(self.content_top, info_h)
        self.screen = pygame.display.set_mode((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.TILE = BOARD_SIZE // 3

        self.state: tuple[int, ...] = GOAL
        self.goal_state = GOAL
        self.mode = Mode.PLAY
        self.algo = SearchAlgo.BFS
        self.search_result: SearchResult | None = None
        self.search_steps: list[SearchStep] = []
        self.step_index = 0
        self.solution_index = 0
        self.anim_timer = 0.0
        self.playing = False
        self.status_msg = "Chơi thủ công — chọn thuật toán rồi bấm Giải."
        self.log_history: list[str] = []
        self.live_reached = 0
        self.live_frontier = 0
        self.snap_timer = 0.0
        self.max_reached = 1
        self.max_frontier = 1
        self.hover_tile: int | None = None

        self.edit_mode = False
        self.edit_buffer = ""
        self.edit_error = ""
        self.edit_cursor_on = True
        self.edit_blink = 0.0

        self.slide_anim: SlideAnim | None = None
        self.pending_state: tuple[int, ...] | None = None

        self._build_buttons()

    def _ensure_solvable(self, state: tuple[int, ...]) -> tuple[tuple[int, ...], str]:
        fixed, changed = make_solvable(state)
        if not changed:
            return fixed, ""
        return fixed, (
            f"Đã chỉnh parity (đảo {inversion_count(state)}→{inversion_count(fixed)}). "
        )

    def _current_step(self) -> SearchStep | None:
        if self.mode == Mode.VISUALIZE and self.search_steps and self.step_index > 0:
            return self.search_steps[self.step_index - 1]
        return None

    def _sync_bar_maxima(self) -> None:
        if not self.search_steps:
            self.max_reached = 1
            self.max_frontier = 1
            return
        self.max_reached = max(max(s.reached for s in self.search_steps), 1)
        self.max_frontier = max(max(s.frontier for s in self.search_steps), 1)

    def _reached_display_count(self) -> int:
        cur = self._current_step()
        if cur is not None:
            return cur.reached
        if self.search_result and self.algo is not SearchAlgo.IDS:
            return self.max_reached
        return self.live_reached

    def _frontier_display_count(self) -> int:
        cur = self._current_step()
        if cur is not None:
            return cur.frontier
        if self.search_result:
            return self.max_frontier
        return self.live_frontier

    def _depth_limit_display(self) -> int:
        cur = self._current_step()
        if cur and cur.depth_limit is not None:
            return cur.depth_limit
        if self.search_result and self.search_result.final_depth_limit is not None:
            return self.search_result.final_depth_limit
        return 1

    def _depth_display(self) -> int:
        cur = self._current_step()
        return cur.depth if cur else 0

    def _metric_pair(self, value: int, maximum: int) -> str:
        return f"{value}/{maximum}"

    def _info_lines(self) -> list[str]:
        valid = is_solvable(self.state)
        struct = {
            SearchAlgo.BFS: "FIFO Queue",
            SearchAlgo.DFS: "Stack LIFO",
            SearchAlgo.IDS: "Stack DLS",
            SearchAlgo.UCS: "PQ theo g(n)",
            SearchAlgo.GREEDY: "PQ theo h(n)",
            SearchAlgo.ASTAR: "PQ theo f(n)=g(n)+h(n)",
            SearchAlgo.IDASTAR: "Stack DFS + threshold f(n)",
        }[self.algo]
        lines = [
            f"Thuật toán: {self.algo.value}",
            f"Chế độ: {self._mode_label()}",
            f"Ma trận: {'Hợp lệ' if valid else 'Không hợp lệ'}",
            f"Cấu trúc: {struct}",
        ]

        searching = self.search_result and self.mode in (Mode.VISUALIZE, Mode.SOLUTION)
        if searching:
            if self.algo in (SearchAlgo.IDS, SearchAlgo.IDASTAR):
                lim = self._depth_limit_display()
                lines.append(f"|stack|: {self._metric_pair(self._frontier_display_count(), self.max_frontier)}")
                lines.append(f"Độ sâu/limit: {self._metric_pair(self._depth_display(), lim)}")
                cur = self._current_step()
                rnd = cur.ids_round if cur and cur.ids_round is not None else (
                    self.search_result.ids_rounds if self.search_result else 0
                )
                lines.append(f"{'Vòng IDS' if self.algo is SearchAlgo.IDS else 'Vòng IDA*'}: {rnd}")
            else:
                lines.append(
                    f"|reached|: {self._metric_pair(self._reached_display_count(), self.max_reached)}"
                )
                lines.append(
                    f"|frontier|: {self._metric_pair(self._frontier_display_count(), self.max_frontier)}"
                )

            if self.mode == Mode.VISUALIZE and self.search_steps:
                lines.append(
                    f"Bước log: {self._metric_pair(self.step_index, len(self.search_steps))}"
                )
            if self.search_result:
                lines.append(f"Nút mở rộng: {self.search_result.nodes_expanded}")
                lines.append(f"Lời giải: {len(self.search_result.path) - 1} trượt")
            if self.mode == Mode.SOLUTION and self.search_result:
                total = len(self.search_result.path) - 1
                lines.append(f"Phát lời giải: {self._metric_pair(self.solution_index, total)}")

        return lines

    def _append_log(self, line: str) -> None:
        self.log_history.append(line)
        if len(self.log_history) > LOG_HISTORY_MAX:
            self.log_history = self.log_history[-LOG_HISTORY_MAX:]

    def _can_manual_play(self) -> bool:
        """Chơi tay khi bảng đứng yên (không đang chạy BFS/lời giải)."""
        if self.edit_mode or self.slide_anim or self.snap_timer > 0:
            return False
        if self.playing and self.mode in (Mode.VISUALIZE, Mode.SOLUTION):
            return False
        return True

    def _build_buttons(self) -> None:
        """Lưới 4 hàng: thuật toán | heuristic | hành động | phát."""
        rw = self.W - RIGHT_X - MARGIN
        x = RIGHT_X
        y = HEADER_H + 8
        g = BTN_GAP
        h = BTN_ROW_H

        fourth = (rw - 3 * g) // 4
        self.btn_bfs = Button(pygame.Rect(x, y, fourth, h), "BFS", toggle=True)
        self.btn_dfs = Button(pygame.Rect(x + fourth + g, y, fourth, h), "DFS", toggle=True)
        self.btn_ids = Button(pygame.Rect(x + 2 * (fourth + g), y, fourth, h), "IDS", toggle=True)
        self.btn_ucs = Button(pygame.Rect(x + 3 * (fourth + g), y, fourth, h), "UCS", toggle=True)
        self.btn_bfs.selected = True
        y += h + g

        third = (rw - 2 * g) // 3
        self.btn_greedy = Button(pygame.Rect(x, y, third, h), "Greedy", toggle=True)
        self.btn_astar = Button(pygame.Rect(x + third + g, y, third, h), "A*", toggle=True)
        self.btn_idastar = Button(pygame.Rect(x + 2 * (third + g), y, third, h), "IDA*", toggle=True)
        y += h + g

        fourth = (rw - 3 * g) // 4
        self.btn_shuffle = Button(pygame.Rect(x, y, fourth, h), "Xáo trộn")
        self.btn_reset = Button(pygame.Rect(x + fourth + g, y, fourth, h), "Đặt lại")
        self.btn_solve = Button(pygame.Rect(x + 2 * (fourth + g), y, fourth, h), "Giải")
        self.btn_edit = Button(pygame.Rect(x + 3 * (fourth + g), y, fourth, h), "Nhập")
        y += h + g

        quad = (rw - 3 * g) // 4
        self.btn_play = Button(pygame.Rect(x, y, quad, h), "Phát")
        self.btn_pause = Button(pygame.Rect(x + quad + g, y, quad, h), "Dừng")
        self.btn_step = Button(pygame.Rect(x + 2 * (quad + g), y, quad, h), "+1")
        self.btn_solution = Button(pygame.Rect(x + 3 * (quad + g), y, quad, h), "Lời giải")

    def shuffle(self) -> None:
        rng = random.Random()
        tiles = list(GOAL)
        for _ in range(120):
            tiles = list(rng.choice(neighbors(tuple(tiles))))
        self._set_state_animated(tuple(tiles), force=True)
        self._reset_run()
        self.status_msg = "Đã xáo trộn (luôn có lời giải)."

    def _reset_run(self) -> None:
        self.mode = Mode.PLAY
        self.search_result = None
        self.search_steps = []
        self.step_index = 0
        self.solution_index = 0
        self.playing = False
        self.anim_timer = 0.0
        self.log_history = []
        self.live_reached = 0
        self.live_frontier = 0
        self.snap_timer = 0.0
        self.slide_anim = None
        self.pending_state = None

    def _cell_center(self, idx: int) -> tuple[float, float]:
        r, c = divmod(idx, 3)
        t = self.TILE
        return BOARD_X + c * t + t / 2, BOARD_Y + r * t + t / 2

    def _cell_rect(self, idx: int) -> pygame.Rect:
        r, c = divmod(idx, 3)
        t = self.TILE
        return pygame.Rect(BOARD_X + c * t + 4, BOARD_Y + r * t + 4, t - 8, t - 8)

    def _ease(self, t: float) -> float:
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

    def _set_state_animated(self, new_state: tuple[int, ...], *, force: bool = False) -> None:
        if self.slide_anim is not None:
            self.state = self.pending_state or self.state
            self.slide_anim = None
            self.pending_state = None

        if not force and self.state != new_state:
            move = slide_move(self.state, new_state)
            if move is not None:
                val, fi, ti = move
                self.pending_state = new_state
                self.slide_anim = SlideAnim(val, fi, ti, 0.0)
                return

        self.state = new_state
        self.pending_state = None
        self.slide_anim = None

    def _open_edit_mode(self) -> None:
        self.edit_mode = True
        self.edit_buffer = "".join(str(x) for x in self.state)
        self.edit_error = ""
        self.playing = False
        self.status_msg = "Nhập 9 số (0–8), 0 = ô trống. Enter: áp dụng · Esc: hủy"

    def _close_edit_mode(self, *, apply: bool) -> None:
        if apply:
            parsed, err = parse_custom_state(self.edit_buffer)
            if parsed is None:
                self.edit_error = err
                return
            fixed, note = self._ensure_solvable(parsed)
            self._set_state_animated(fixed, force=True)
            self._reset_run()
            self.status_msg = f"Đã áp dụng ma trận. {note}".strip()
        self.edit_mode = False
        self.edit_error = ""

    def _edit_handle_key(self, event: pygame.event.Event) -> None:
        if event.key == pygame.K_RETURN:
            self._close_edit_mode(apply=True)
        elif event.key == pygame.K_ESCAPE:
            self._close_edit_mode(apply=False)
            self.status_msg = "Đã hủy nhập ma trận."
        elif event.key == pygame.K_BACKSPACE:
            self.edit_buffer = self.edit_buffer[:-1]
            self.edit_error = ""
        elif event.unicode and event.unicode.isdigit() and len(self.edit_buffer) < 18:
            self.edit_buffer += event.unicode
            self.edit_error = ""

    def run_solve(self) -> None:
        if self.slide_anim:
            self.state = self.pending_state or self.state
            self.slide_anim = None
            self.pending_state = None

        start, note = self._ensure_solvable(self.state)
        if start != self.state:
            self.state = start

        result = solve(start, self.algo, record_steps=True)
        if result is None:
            self.status_msg = f"{self.algo.value} thất bại."
            return
        path = result.path
        self.search_result = result
        self.search_steps = result.steps
        self.step_index = 0
        self.solution_index = 0
        self.state = start
        self.slide_anim = None
        self.pending_state = None
        self.snap_timer = 0.0
        self.log_history = []
        if self.search_steps:
            self.live_reached = self.search_steps[0].reached
            self.live_frontier = self.search_steps[0].frontier
        else:
            self.live_reached = 0
            self.live_frontier = 0
        self._sync_bar_maxima()
        self.mode = Mode.VISUALIZE
        self.playing = True
        n_moves = len(path) - 1
        algo = self.algo.value
        msg = f"{algo} — {result.nodes_expanded} nút mở rộng, lời giải {n_moves} bước."
        self.status_msg = (note + msg) if note else msg

    def _log_finished(self) -> bool:
        return self.step_index >= len(self.search_steps)

    def _path_finished(self) -> bool:
        if not self.search_result:
            return True
        return self.solution_index >= len(self.search_result.path) - 1

    def _show_board_state(self, new_state: tuple[int, ...]) -> bool:
        """Cập nhật bảng; trả về True nếu đang trượt (chờ animation)."""
        if self.state == new_state:
            return False
        move = slide_move(self.state, new_state)
        if move is not None:
            val, fi, ti = move
            self.pending_state = new_state
            self.slide_anim = SlideAnim(val, fi, ti, 0.0)
            return True
        self.state = new_state
        self.snap_timer = SNAP_STEP_DELAY
        return False

    def advance_search_step(self) -> None:
        """Một bước tìm kiếm: log + đồng bộ bảng theo trạng thái thuật toán."""
        if self._log_finished():
            return
        step = self.search_steps[self.step_index]
        self._append_log(format_search_log(step, self.algo))
        self.live_reached = step.reached
        self.live_frontier = step.frontier
        self._show_board_state(step.state)
        self.step_index += 1
        if self._log_finished():
            tag = self.algo.value
            self._append_log(f"[{tag}] Kết thúc tìm kiếm — bấm 'Lời giải' để xem đường đi.")
            self.playing = False
            self.status_msg = f"{tag} hoàn tất. Bấm 'Lời giải' để phát animation đường đi."

    def advance_solution_step(self) -> None:
        if not self.search_result or self.slide_anim is not None or self._path_finished():
            return
        path = self.search_result.path
        nxt = self.solution_index + 1
        total = len(path) - 1
        move = slide_move(self.state, path[nxt])
        if move is None:
            self.state = path[nxt]
            self.solution_index = nxt
            self._append_log(f"[Lời giải] Bước {nxt}/{total} — chuyển trạng thái")
            return
        val, fi, ti = move
        self.pending_state = path[nxt]
        self.slide_anim = SlideAnim(val, fi, ti, 0.0)
        self.solution_index = nxt
        self._append_log(f"[Lời giải] Bước {nxt}/{total} — trượt ô {val}")

    def _begin_solution_playback(self) -> None:
        if not self.search_result or not self.search_result.path:
            return
        path = self.search_result.path
        self.mode = Mode.SOLUTION
        self.solution_index = 0
        self.state = path[0]
        self.slide_anim = None
        self.pending_state = None
        self.snap_timer = 0.0
        self.log_history = []
        self.playing = True
        total = len(path) - 1
        self._append_log(f"[Lời giải] Bắt đầu phát — {total} bước trượt (0.75s/bước)")
        self.status_msg = "Đang phát lời giải trên bảng."

    def try_move_tile(self, index: int) -> None:
        if not self._can_manual_play():
            return
        bi = blank_index(self.state)
        br, bc = divmod(bi, 3)
        tr, tc = divmod(index, 3)
        if abs(br - tr) + abs(bc - tc) != 1:
            return
        board = list(self.state)
        board[bi], board[index] = board[index], board[bi]
        new_state = tuple(board)
        self._set_state_animated(new_state, force=True)
        self.playing = False
        self.mode = Mode.PLAY
        self.search_result = None
        self.search_steps = []
        self.step_index = 0
        self.log_history = []
        if new_state == GOAL:
            self.status_msg = "Chúc mừng — bạn đã xếp xong!"
        else:
            self.status_msg = f"Chơi tay — bấm Giải để chạy {self.algo.value}."

    def tile_at_pos(self, pos: tuple[int, int]) -> int | None:
        x, y = pos
        if not (BOARD_X <= x < BOARD_X + BOARD_SIZE and BOARD_Y <= y < BOARD_Y + BOARD_SIZE):
            return None
        return (y - BOARD_Y) // self.TILE * 3 + (x - BOARD_X) // self.TILE

    def draw_header(self) -> None:
        pygame.draw.line(
            self.screen, COLORS["section_border"], (MARGIN, HEADER_H), (self.W - MARGIN, HEADER_H), 1
        )
        self.screen.blit(
            self.font_title.render("8-Puzzle · BFS / DFS / IDS / UCS / Greedy / A* / IDA*", True, COLORS["title"]),
            (MARGIN, 14),
        )
        self.screen.blit(
            self.font_body.render(
                "Nhấp ô kề ô trống để chơi tay (khi bảng dừng)  ·  E: nhập  ·  Space: phát/dừng",
                True,
                COLORS["muted"],
            ),
            (MARGIN, 42),
        )

    def _tile_color(self, val: int, idx: int, highlight: str | None) -> tuple[int, int, int]:
        if val == 0:
            return COLORS["tile_blank"]
        bi = blank_index(self.pending_state or self.state)
        if highlight and idx == bi:
            return COLORS["highlight_dequeue"] if highlight == "dequeue" else COLORS["highlight_enqueue"]
        if self.goal_state[idx] == val:
            return COLORS["tile_goal"]
        if self.hover_tile == idx and self._can_manual_play():
            return COLORS["tile_hover"]
        return COLORS["tile"]

    def _draw_tile_at(self, val: int, cx: float, cy: float, idx_for_color: int, highlight: str | None) -> None:
        t = self.TILE
        size = t - 8
        rect = pygame.Rect(0, 0, size, size)
        rect.center = (int(cx), int(cy))
        col = self._tile_color(val, idx_for_color, highlight)
        pygame.draw.rect(self.screen, col, rect, border_radius=12)
        pygame.draw.rect(self.screen, COLORS["section_border"], rect, 2, border_radius=12)
        if val:
            txt = self.font_tile.render(str(val), True, COLORS["tile_num"])
            self.screen.blit(txt, txt.get_rect(center=rect.center))

    def _board_logical_state(self) -> tuple[int, ...]:
        if self.edit_mode:
            preview, _ = parse_custom_state(self.edit_buffer)
            if preview is not None:
                return preview
        return self.state

    def draw_board(self) -> None:
        highlight: str | None = None
        if self.mode == Mode.VISUALIZE and self.search_steps and self.step_index > 0:
            last = self.search_steps[self.step_index - 1]
            if last.kind is StepKind.REMOVE:
                highlight = "dequeue"
            elif last.kind is StepKind.ADD:
                highlight = "enqueue"

        logical = self._board_logical_state()
        display = self.pending_state if self.slide_anim and self.pending_state else logical
        sliding_val = self.slide_anim.value if self.slide_anim else -1

        for i in range(9):
            val = display[i]
            if val == sliding_val:
                pygame.draw.rect(self.screen, COLORS["tile_blank"], self._cell_rect(i), border_radius=12)
                continue
            self._draw_tile_at(val, *self._cell_center(i), i, highlight)

        if self.slide_anim:
            a = self.slide_anim
            p = self._ease(a.t)
            x0, y0 = self._cell_center(a.from_idx)
            x1, y1 = self._cell_center(a.to_idx)
            cx = x0 + (x1 - x0) * p
            cy = y0 + (y1 - y0) * p
            self._draw_tile_at(a.value, cx, cy, a.to_idx, highlight)

        frame = pygame.Rect(BOARD_X - 5, BOARD_Y - 5, BOARD_SIZE + 10, BOARD_SIZE + 10)
        pygame.draw.rect(self.screen, COLORS["accent_dim"], frame, 2, border_radius=14)

    def draw_edit_overlay(self) -> None:
        if not self.edit_mode:
            return
        box = pygame.Rect(BOARD_X - 8, BOARD_Y - 8, BOARD_SIZE + 16, BOARD_SIZE + 16)
        overlay = pygame.Surface((box.width, box.height), pygame.SRCALPHA)
        overlay.fill(COLORS["overlay"])
        self.screen.blit(overlay, box.topleft)

        panel = pygame.Rect(BOARD_X, BOARD_Y + BOARD_SIZE // 2 - 70, BOARD_SIZE, 140)
        pygame.draw.rect(self.screen, COLORS["input_bg"], panel, border_radius=10)
        pygame.draw.rect(self.screen, COLORS["input_border"], panel, 2, border_radius=10)

        y = panel.y + 12
        for line in (
            "Nhập ma trận (0 = ô trống)",
            "VD: 123456780",
            "hoặc: 1 2 3 4 5 6 7 8 0",
        ):
            self.screen.blit(self.font_sm.render(line, True, COLORS["muted"]), (panel.x + 12, y))
            y += 18

        field = pygame.Rect(panel.x + 12, y, panel.width - 24, 32)
        pygame.draw.rect(self.screen, (14, 18, 30), field, border_radius=6)
        pygame.draw.rect(self.screen, COLORS["input_border"], field, 1, border_radius=6)
        shown = self.edit_buffer + ("|" if self.edit_cursor_on else "")
        self.screen.blit(self.font_sm.render(shown, True, COLORS["tile_num"]), (field.x + 8, field.y + 8))
        y = field.bottom + 8
        if self.edit_error:
            self.screen.blit(
                self.font_sm.render(self.edit_error, True, COLORS["input_error"]),
                (panel.x + 12, y),
            )
        else:
            self.screen.blit(
                self.font_sm.render("Enter: áp dụng  ·  Esc: hủy", True, COLORS["muted"]),
                (panel.x + 12, y),
            )

    def draw_controls_frame(self) -> None:
        """Khung nhẹ quanh vùng nút."""
        rw = self.W - RIGHT_X - MARGIN
        box = pygame.Rect(
            RIGHT_X - 4,
            HEADER_H + 4,
            rw + 8,
            calc_controls_bottom() - HEADER_H - 4,
        )
        pygame.draw.rect(self.screen, COLORS["section"], box, border_radius=10)
        pygame.draw.rect(self.screen, COLORS["section_border"], box, 1, border_radius=10)

    def draw_left_legend(self) -> None:
        legend_y = BOARD_Y + BOARD_SIZE + 16
        legend_h = section_title_height(self.font_heading) + 2 * LEGEND_ROW_H + SECTION_PAD
        box = pygame.Rect(MARGIN, legend_y, LEFT_W, legend_h)
        inner_y = draw_section_box(self.screen, box, "Chú thích màu", self.font_heading)
        row_x = box.x + SECTION_PAD
        row_y = inner_y
        items = [
            (COLORS["tile_goal"], "Số đúng vị trí đích"),
            (COLORS["tile_hover"], "Ô có thể nhấp (chơi tay)"),
        ]
        for col, label in items:
            row_y = draw_legend_row(self.screen, row_x, row_y, col, label, self.font_sm)

    def draw_right_panel(self) -> None:
        self.draw_controls_frame()
        for btn in (
            self.btn_bfs,
            self.btn_dfs,
            self.btn_ids,
            self.btn_ucs,
            self.btn_greedy,
            self.btn_astar,
            self.btn_idastar,
            self.btn_shuffle,
            self.btn_reset,
            self.btn_solve,
            self.btn_edit,
            self.btn_play,
            self.btn_pause,
            self.btn_step,
            self.btn_solution,
        ):
            btn.draw(self.screen, self.font_btn)

        y = self.content_top
        pad = SECTION_PAD
        inner_w = self.W - RIGHT_X - MARGIN - 2 * pad
        rw = self.W - RIGHT_X - MARGIN

        info_lines = self._info_lines()
        info_h = calc_info_height(self.font_sm, len(info_lines))
        info_rect = pygame.Rect(RIGHT_X, y, rw, info_h)
        inner_y = draw_section_box(self.screen, info_rect, "Thông tin", self.font_heading)
        col_gap = 10
        col_w = (inner_w - col_gap) // 2
        clip = self.screen.get_clip()
        self.screen.set_clip(info_rect.inflate(-2, -2))
        draw_text_columns(
            self.screen,
            info_rect.x + pad,
            inner_y,
            col_w + col_gap,
            info_lines,
            self.font_sm,
            line_gap=4,
        )
        self.screen.set_clip(clip)
        y = info_rect.bottom + SECTION_GAP

        bar_rect = pygame.Rect(RIGHT_X, y, rw, BAR_SECTION_H)
        bar_title = {
            SearchAlgo.BFS: "Reached & Frontier (BFS)",
            SearchAlgo.DFS: "Reached & Stack (DFS)",
            SearchAlgo.IDS: "Depth & Stack (IDS)",
            SearchAlgo.UCS: "Reached & PQ (UCS)",
            SearchAlgo.GREEDY: "Reached & PQ (Greedy)",
            SearchAlgo.ASTAR: "Reached & PQ (A*)",
            SearchAlgo.IDASTAR: "Depth & Stack (IDA*)",
        }[self.algo]
        inner_y = draw_section_box(self.screen, bar_rect, bar_title, self.font_heading)
        bx = bar_rect.x + pad
        by = inner_y

        if self.algo in (SearchAlgo.IDS, SearchAlgo.IDASTAR):
            by = draw_metric_bar(
                self.screen,
                bx,
                by,
                inner_w,
                "Độ sâu / limit",
                self._depth_display(),
                COLORS["bar_explored"],
                self._depth_limit_display(),
                self.font_sm,
            )
            draw_metric_bar(
                self.screen,
                bx,
                by,
                inner_w,
                "|stack| (DLS)",
                self._frontier_display_count(),
                COLORS["bar_frontier"],
                self.max_frontier,
                self.font_sm,
            )
        else:
            by = draw_metric_bar(
                self.screen,
                bx,
                by,
                inner_w,
                "|reached|",
                self._reached_display_count(),
                COLORS["bar_explored"],
                self.max_reached,
                self.font_sm,
            )
            frontier_labels = {
                SearchAlgo.BFS: "Frontier (FIFO)",
                SearchAlgo.DFS: "Stack (LIFO)",
                SearchAlgo.UCS: "PQ theo g(n)",
                SearchAlgo.GREEDY: "PQ theo h(n)",
                SearchAlgo.ASTAR: "PQ theo f(n)=g+h",
            }
            frontier_label = frontier_labels[self.algo]
            draw_metric_bar(
                self.screen,
                bx,
                by,
                inner_w,
                frontier_label,
                self._frontier_display_count(),
                COLORS["bar_frontier"],
                self.max_frontier,
                self.font_sm,
            )
        y = bar_rect.bottom + SECTION_GAP

        algo_rect = pygame.Rect(RIGHT_X, y, rw, ALGO_SECTION_H)
        inner_y = draw_section_box(self.screen, algo_rect, "Mã giả (tóm tắt)", self.font_heading)
        flow = TextFlow(algo_rect.x + pad, inner_y, inner_w, line_gap=4)
        algo_blurbs = {
            SearchAlgo.BFS: (
                "BFS: frontier FIFO; reached←{INITIAL}; REMOVE; "
                "nếu s NOT IN reached thì reached + {s} và INSERT."
            ),
            SearchAlgo.DFS: (
                "DFS: frontier STACK; reached←{INITIAL}; POP; "
                "nếu s NOT IN reached thì reached + {s} và PUSH."
            ),
            SearchAlgo.IDS: (
                "IDS: for depth=0..∞ gọi DLS(limit). DLS: stack POP; "
                "nếu depth≥limit→cutoff; else mở rộng nếu không cycle."
            ),
            SearchAlgo.UCS: (
                "UCS: PQ theo g(n); chi phí bước = ô số swap với 0; "
                "reached lưu g nhỏ nhất; cập nhật nếu g mới nhỏ hơn."
            ),
            SearchAlgo.GREEDY: (
                "Greedy: PQ theo h(n)=Manhattan; reached khi INSERT; "
                "không tái mở trạng thái đã reached."
            ),
            SearchAlgo.ASTAR: (
                "A*: PQ theo f(n)=g(n)+h(n); g là số bước tích lũy, "
                "h là Manhattan; cập nhật nếu tìm được g nhỏ hơn."
            ),
            SearchAlgo.IDASTAR: (
                "IDA*: threshold khởi tạo = f(start); DFS theo contour f<=threshold; "
                "nếu không thấy đích thì tăng threshold = min f vượt ngưỡng."
            ),
        }
        flow.wrap(self.screen, self.font_sm, algo_blurbs[self.algo])
        y = algo_rect.bottom + SECTION_GAP

        log_rect = pygame.Rect(RIGHT_X, y, rw, LOG_SECTION_H)
        inner_y = draw_section_box(self.screen, log_rect, "Nhật ký chạy", self.font_heading)
        flow = TextFlow(log_rect.x + pad, inner_y, inner_w, line_gap=3)
        clip = self.screen.get_clip()
        self.screen.set_clip(log_rect.inflate(-4, -4))
        visible = self.log_history[-LOG_VISIBLE_LINES:]
        if not visible:
            flow.line(self.screen, self.font_sm, "Chưa có dòng log.", COLORS["muted"])
        else:
            current = self.log_history[-1] if self.log_history else ""
            animating = self.playing or self.slide_anim is not None or self.snap_timer > 0
            for line in visible:
                is_active = animating and line == current
                if is_active:
                    row_y = flow.cursor
                    row_h = self.font_sm.get_height() + 2
                    pygame.draw.rect(
                        self.screen,
                        COLORS["log_active_bg"],
                        (log_rect.x + pad - 2, row_y - 1, inner_w + 4, row_h),
                        border_radius=4,
                    )
                    text = f"▶ {line}"
                    flow.line(self.screen, self.font_sm, text, COLORS["log_active"])
                else:
                    col = COLORS["accent"] if line.startswith("[Lời giải]") else COLORS["text"]
                    if line.startswith("[") and "đích" in line.lower():
                        col = COLORS["success"]
                    flow.line(self.screen, self.font_sm, line, col)
        self.screen.set_clip(clip)

    def _mode_label(self) -> str:
        if self.edit_mode:
            return "Nhập ma trận"
        if self.slide_anim:
            return "Đang trượt ô"
        if self.mode == Mode.VISUALIZE and self.playing:
            return f"{self.algo.value} đang chạy"
        if self.mode == Mode.VISUALIZE and self.search_result:
            return f"{self.algo.value} (tạm dừng / xong)"
        return {
            Mode.PLAY: "Chơi tay",
            Mode.VISUALIZE: f"Trực quan {self.algo.value}",
            Mode.SOLUTION: "Phát lời giải",
        }[self.mode]

    def _select_algo(self, algo: SearchAlgo) -> None:
        self.algo = algo
        self.btn_bfs.selected = algo is SearchAlgo.BFS
        self.btn_dfs.selected = algo is SearchAlgo.DFS
        self.btn_ids.selected = algo is SearchAlgo.IDS
        self.btn_ucs.selected = algo is SearchAlgo.UCS
        self.btn_greedy.selected = algo is SearchAlgo.GREEDY
        self.btn_astar.selected = algo is SearchAlgo.ASTAR
        self.btn_idastar.selected = algo is SearchAlgo.IDASTAR
        self._reset_run()
        self.status_msg = f"Đã chọn {algo.value}."

    def handle_click(self, pos: tuple[int, int]) -> None:
        if self.btn_bfs.hit(pos):
            self._select_algo(SearchAlgo.BFS)
        elif self.btn_dfs.hit(pos):
            self._select_algo(SearchAlgo.DFS)
        elif self.btn_ids.hit(pos):
            self._select_algo(SearchAlgo.IDS)
        elif self.btn_ucs.hit(pos):
            self._select_algo(SearchAlgo.UCS)
        elif self.btn_greedy.hit(pos):
            self._select_algo(SearchAlgo.GREEDY)
        elif self.btn_astar.hit(pos):
            self._select_algo(SearchAlgo.ASTAR)
        elif self.btn_idastar.hit(pos):
            self._select_algo(SearchAlgo.IDASTAR)
        elif self.btn_shuffle.hit(pos):
            self.shuffle()
        elif self.btn_edit.hit(pos):
            self._open_edit_mode()
        elif self.btn_reset.hit(pos):
            self._set_state_animated(GOAL, force=True)
            self._reset_run()
            self.status_msg = "Đã đặt lại trạng thái đích."
        elif self.btn_solve.hit(pos):
            self.run_solve()
        elif self.btn_play.hit(pos):
            self.playing = True
        elif self.btn_pause.hit(pos):
            self.playing = False
        elif self.btn_step.hit(pos):
            if self.slide_anim or self.snap_timer > 0:
                pass
            elif self.mode == Mode.VISUALIZE and self.search_result and not self._log_finished():
                self.advance_search_step()
            elif self.mode == Mode.SOLUTION and self.search_result and not self._path_finished():
                self.advance_solution_step()
            elif self.mode == Mode.PLAY:
                self.run_solve()
        elif self.btn_solution.hit(pos) and self.search_result:
            self._begin_solution_playback()

        if self._can_manual_play():
            idx = self.tile_at_pos(pos)
            if idx is not None:
                self.try_move_tile(idx)

    def _on_slide_finished(self) -> None:
        if not self.playing:
            return
        if self.mode == Mode.VISUALIZE and not self._log_finished():
            self.advance_search_step()
        elif self.mode == Mode.SOLUTION and not self._path_finished():
            self.advance_solution_step()
        elif self.mode == Mode.SOLUTION and self._path_finished():
            self.playing = False
            self._append_log("[Lời giải] Hoàn tất — đã tới trạng thái đích.")
            self.status_msg = "Hoàn thành phát lời giải."

    def update(self, dt: float) -> None:
        if self.edit_mode:
            self.edit_blink += dt
            if self.edit_blink >= 0.5:
                self.edit_blink = 0.0
                self.edit_cursor_on = not self.edit_cursor_on
            return

        if self.slide_anim:
            self.slide_anim.t += dt / SLIDE_DURATION
            if self.slide_anim.t >= 1.0:
                self.state = self.pending_state or self.state
                self.slide_anim = None
                self.pending_state = None
                self._on_slide_finished()
            return

        if self.snap_timer > 0:
            self.snap_timer = max(0.0, self.snap_timer - dt)
            if self.snap_timer > 0:
                return

        if not self.playing:
            return

        if self.mode == Mode.VISUALIZE and not self._log_finished():
            self.advance_search_step()
        elif self.mode == Mode.SOLUTION and not self._path_finished():
            self.advance_solution_step()

    def run(self) -> None:
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if self.edit_mode:
                        self._edit_handle_key(event)
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_e:
                        self._open_edit_mode()
                    elif event.key == pygame.K_SPACE:
                        self.playing = not self.playing
                    elif event.key == pygame.K_s:
                        self.shuffle()
                    elif event.key == pygame.K_r:
                        self._set_state_animated(GOAL, force=True)
                        self._reset_run()
                elif event.type == pygame.MOUSEMOTION:
                    self.hover_tile = self.tile_at_pos(event.pos)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)

            self.update(dt)
            self.screen.fill(COLORS["bg"])
            self.draw_header()
            self.draw_board()
            self.draw_edit_overlay()
            self.draw_left_legend()
            self.draw_right_panel()
            pygame.display.flip()

        pygame.quit()
        sys.exit(0)


def main() -> None:
    PuzzleApp().run()


if __name__ == "__main__":
    main()
