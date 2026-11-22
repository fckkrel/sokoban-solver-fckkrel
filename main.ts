// Sokoban solver — push-optimal, room by room.
//
// Idea: a level is a 1-wide hallway with small rooms off it. A box pushed out
// of a room through its door can never come back, so each room is its own
// problem and the level optimum is the sum of the rooms' optima. I find the
// rooms by cutting the hallway's dead squares out of the floor (a box standing
// in the hallway can never reach a goal), then solve every room with an A*
// search over pushes. The search state is the boxes plus the player's
// reachable region; it prunes dead squares and frozen 2x2 blocks, and its
// heuristic is the cheapest assignment of boxes to goals by walking distance,
// which never overestimates (one push moves one box one square), so the first
// solution A* pops is push-optimal. Everything is budgeted by work (a state
// cap), never by the clock, so two runs answer byte for byte alike.
//
// I/O: reads /input/<id>/level.txt for every instance directory, writes
// /output/<id>/solution.txt (LURD moves, an upper letter marks a push).

import { mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Most states one room search may keep (work, never the clock).
const STATES_PER_ROOM = 400_000;

// The four moves: [dx, dy, step letter, push letter].
const DIRS = [
  { dx: 0, dy: -1, step: 'u', push: 'U' },
  { dx: 0, dy: 1, step: 'd', push: 'D' },
  { dx: -1, dy: 0, step: 'l', push: 'L' },
  { dx: 1, dy: 0, step: 'r', push: 'R' },
] as const;

type Level = {
  w: number;
  h: number;
  wall: Uint8Array; // 1 = wall
  goals: number[];
  boxes: number[];
  player: number;
};

/** Parse XSB text (# wall, space floor, . goal, $ box, * box on goal, @ player, + player on goal). */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/**
 * Squares from which a box alone can be pushed onto some goal: pull back from
 * the goals over open floor. A box on any other square can never reach a goal,
 * whatever the other boxes do, so it is safe to prune.
 */
function liveCells(lv: Level): Uint8Array {
  const { w, wall, goals } = lv;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (let k = 0; k < 4; k++) {
      const d = DIRS[k].dy * w + DIRS[k].dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/**
 * Flood the squares the player reaches from `from` with `occupied` in the way,
 * marking them in `seen` with `stamp`; returns the lowest square reached,
 * which names the region (the player's exact square inside it is free).
 */
function reach(
  wall: Uint8Array,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
  w: number,
): number {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = p + (k === 0 ? -w : k === 1 ? w : k === 2 ? -1 : 1);
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** Walking distance from every goal to every square, around walls alone. */
function goalDistances(lv: Level): Int32Array[] {
  const { w, wall, goals } = lv;
  const out: Int32Array[] = [];
  for (const g of goals) {
    const dist = new Int32Array(wall.length).fill(-1);
    dist[g] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const p = queue[qi];
      for (let k = 0; k < 4; k++) {
        const q = p + (k === 0 ? -w : k === 1 ? w : k === 2 ? -1 : 1);
        if (!wall[q] && dist[q] < 0) {
          dist[q] = dist[p] + 1;
          queue.push(q);
        }
      }
    }
    out.push(dist);
  }
  return out;
}

/**
 * Cheapest assignment of boxes to goals by walking distance (bitmask DP over
 * goal subsets). Never overestimates the pushes left: one push moves one box
 * one square, so the distances can shrink by at most one per push. Returns
 * Infinity when no box-to-goal matching exists (the state is dead).
 */
function assignmentHeuristic(boxes: number[], dist: Int32Array[]): number {
  const n = boxes.length;
  if (n === 0) return 0;
  const full = (1 << n) - 1;
  const dp = new Float64Array(1 << n).fill(Infinity);
  dp[0] = 0;
  for (let mask = 0; mask <= full; mask++) {
    const bi = popcount(mask);
    if (bi >= n || dp[mask] === Infinity) continue;
    for (let g = 0; g < n; g++) {
      if (mask & (1 << g)) continue;
      const d = dist[g][boxes[bi]];
      if (d < 0) continue;
      const nm = mask | (1 << g);
      const v = dp[mask] + d;
      if (v < dp[nm]) dp[nm] = v;
    }
  }
  return dp[full];
}

function popcount(x: number): number {
  let c = 0;
  while (x) {
    x &= x - 1;
    c++;
  }
  return c;
}

/**
 * Whether the box just pushed to `t` sits in a frozen 2x2 block: four squares
 * of walls and boxes, with a box among them off its goal. No box in such a
 * block can move — pushing it needs the player on a square of the block, or
 * its target to be a square of the block, and both are taken — so the state
 * is dead. (A box on a goal inside the block is fine: the block only kills a
 * state when one of its boxes still has somewhere to go.)
 */
function frozenBlock(
  wall: Uint8Array,
  occupied: Uint8Array,
  goal: Uint8Array,
  t: number,
  w: number,
): boolean {
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a])) continue;
    if (!(wall[b] || occupied[b])) continue;
    if (!(wall[c] || occupied[c])) continue;
    if (!(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) {
      return true;
    }
  }
  return false;
}

/** A state's key: its region's lowest square, then the boxes, as digits in base `size`. */
function stateKey(boxes: number[], low: number, size: number): number {
  let k = low;
  for (const b of boxes) k = k * size + b;
  return k;
}

/** A min-heap of state indices, ordered by (f, then deeper g, then older index). */
class Heap {
  private a: number[] = [];
  private f: number[];
  private g: number[];
  constructor(f: number[], g: number[]) {
    this.f = f;
    this.g = g;
  }
  get size(): number {
    return this.a.length;
  }
  push(i: number): void {
    const a = this.a;
    a.push(i);
    let c = a.length - 1;
    while (c > 0) {
      const p = (c - 1) >> 1;
      if (this.less(a[c], a[p])) {
        [a[c], a[p]] = [a[p], a[c]];
        c = p;
      } else break;
    }
  }
  pop(): number {
    const a = this.a;
    const top = a[0];
    const last = a.pop() as number;
    if (a.length > 0) {
      a[0] = last;
      let p = 0;
      for (;;) {
        const l = p * 2 + 1;
        const r = l + 1;
        let m = p;
        if (l < a.length && this.less(a[l], a[m])) m = l;
        if (r < a.length && this.less(a[r], a[m])) m = r;
        if (m === p) break;
        [a[m], a[p]] = [a[p], a[m]];
        p = m;
      }
    }
    return top;
  }
  private less(i: number, j: number): boolean {
    if (this.f[i] !== this.f[j]) return this.f[i] < this.f[j];
    if (this.g[i] !== this.g[j]) return this.g[i] > this.g[j];
    return i < j;
  }
}

/**
 * The push-optimal solution of a level: {pushes, moves} (moves in LURD, an
 * upper letter a push), or null when the search gives up (state cap) or the
 * level has no solution. A* over pushes with the assignment heuristic; the
 * heuristic is admissible and consistent, so the first solved state popped
 * from the heap is optimal.
 */
function solvePush(lv: Level, maxStates: number): { pushes: number; moves: string } | null {
  const { w, wall, goals } = lv;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveCells(lv);
  for (const b of lv.boxes) if (!live[b]) return null;
  const dist = goalDistances(lv);
  const startBoxes = lv.boxes.slice().sort((a, b) => a - b);
  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b]);
  if (solved(startBoxes)) return { pushes: 0, moves: '' };

  const sBoxes: number[][] = [startBoxes];
  const sPlayer: number[] = [lv.player];
  const sParent: number[] = [-1];
  const sFrom: number[] = [-1];
  const sTo: number[] = [-1];
  const sDir: number[] = [-1];
  const sG: number[] = [0];
  const sF: number[] = [assignmentHeuristic(startBoxes, dist)];
  const sKey: number[] = [];
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  const stack2 = new Int32Array(size);
  let stamp = 0;
  const occupied = new Uint8Array(size);

  const heap = new Heap(sF, sG);
  for (const b of startBoxes) occupied[b] = 1;
  const startLow = reach(wall, occupied, lv.player, seen, ++stamp, stack, w);
  for (const b of startBoxes) occupied[b] = 0;
  sKey.push(stateKey(startBoxes, startLow, size));
  const best = new Map<number, number>();
  best.set(sKey[0], 0);
  heap.push(0);

  let found = -1;
  while (heap.size > 0) {
    const si = heap.pop();
    if (sG[si] > (best.get(sKey[si]) ?? Infinity)) continue; // a better copy of this state was queued
    const boxes = sBoxes[si];
    if (solved(boxes)) {
      found = si;
      break;
    }
    const g = sG[si];
    for (const b of boxes) occupied[b] = 1;
    reach(wall, occupied, sPlayer[si], seen, ++stamp, stack, w);
    const here = stamp;
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi];
      for (let di = 0; di < 4; di++) {
        const d = DIRS[di].dy * w + DIRS[di].dx;
        const s = b - d; // the square the player must stand on to push
        const t = b + d; // where the box lands
        if (seen[s] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(wall, occupied, goal, t, w);
        let key = -1;
        let next: number[] | null = null;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((a, b) => a - b);
          const low = reach(wall, occupied, b, mark, ++stamp, stack2, w);
          key = stateKey(next, low, size);
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || next === null) continue;
        const h = assignmentHeuristic(next, dist);
        if (h === Infinity) continue; // no box-to-goal matching left: dead
        const ng = g + 1;
        const prev = best.get(key);
        if (prev !== undefined && prev <= ng) continue;
        if (sBoxes.length >= maxStates) return null;
        best.set(key, ng);
        const ni = sBoxes.length;
        sBoxes.push(next);
        sPlayer.push(b);
        sParent.push(si);
        sFrom.push(b);
        sTo.push(t);
        sDir.push(di);
        sG.push(ng);
        sF.push(ng + h);
        sKey.push(key);
        heap.push(ni);
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;

  // Walk the parents back, then replay the pushes forward with walks between them.
  const path: number[] = [];
  for (let s = found; s > 0; s = sParent[s]) path.push(s);
  path.reverse();
  let moves = '';
  let player = lv.player;
  let boxes = startBoxes;
  for (const s of path) {
    const next = sBoxes[s];
    const from = sFrom[s];
    const di = sDir[s];
    const d = DIRS[di].dy * w + DIRS[di].dx;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(wall, occupied, player, from - d, w);
    for (const b of boxes) occupied[b] = 0;
    moves += DIRS[di].push;
    player = from;
    boxes = next;
  }
  return { pushes: path.length, moves };
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower LURD letters. */
function walkTo(
  wall: Uint8Array,
  occupied: Uint8Array,
  from: number,
  to: number,
  w: number,
): string {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (let k = 0; k < 4; k++) {
      const q = p + (k === 0 ? -w : k === 1 ? w : k === 2 ? -1 : 1);
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = DIRS[k].step;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

/** The square the player stands on after `moves` (case aside) from `from` on a board `w` wide. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway:
 * a box standing there reaches no goal) out of the floor, and each part left
 * that holds a box is a room. Null when the level is no set of rooms — some
 * part holds more or fewer boxes than goals, a goal sits in the cut, or a box
 * starts in the hallway — and the caller falls back to the whole level.
 */
function roomsOf(lv: Level): { boxes: number[]; goals: number[] }[] | null {
  const { w, wall, goals, boxes, player } = lv;
  const live = liveCells(lv);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let x = 0; x < w; x++) {
    const i = row * w + x;
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: { boxes: number[]; goals: number[] }[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const cells = [s];
    part[s] = rooms.length;
    for (let i = 0; i < cells.length; i++) {
      const c = cells[i];
      for (let k = 0; k < 4; k++) {
        const q = c + (k === 0 ? -w : k === 1 ? w : k === 2 ? -1 : 1);
        if (q < 0 || q >= wall.length) continue;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          cells.push(q);
        }
      }
    }
    rooms.push({ boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]].boxes.push(b);
  for (const g of goals) {
    if (part[g] < 0) return null;
    rooms[part[g]].goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/**
 * The level solved room by room: each room's push-optimal solution in turn,
 * the other rooms' boxes standing as walls (on their goals once their room
 * is done), the player walking on from where the last room left it. A box
 * never leaves its room — a push through a door lands it in the hallway,
 * where it reaches no goal — so the rooms' optima add up to the level's.
 * Null when a room's search gives up.
 */
function solveByRooms(lv: Level, maxStates: number): { pushes: number; moves: string } | null {
  const rooms = roomsOf(lv);
  if (rooms === null) return solvePush(lv, maxStates);
  let player = lv.player;
  let pushes = 0;
  let moves = '';
  const done = new Array<boolean>(rooms.length).fill(false);
  for (let ri = 0; ri < rooms.length; ri++) {
    const wall = lv.wall.slice();
    for (let oj = 0; oj < rooms.length; oj++) {
      if (oj === ri) continue;
      const src = done[oj] ? rooms[oj].goals : rooms[oj].boxes;
      for (const b of src) wall[b] = 1;
    }
    const sub: Level = {
      w: lv.w,
      h: lv.h,
      wall,
      goals: rooms[ri].goals.slice().sort((a, b) => a - b),
      boxes: rooms[ri].boxes.slice().sort((a, b) => a - b),
      player,
    };
    const solved = solvePush(sub, maxStates);
    if (solved === null) return null;
    pushes += solved.pushes;
    moves += solved.moves;
    player = endOf(lv.w, player, solved.moves);
    done[ri] = true;
  }
  return { pushes, moves };
}

/** One level's answer: its moves, or '' when it could not be solved. */
function solveLevel(text: string): string {
  const lv = parseLevel(text);
  const byRooms = solveByRooms(lv, STATES_PER_ROOM);
  if (byRooms !== null) return byRooms.moves;
  // Last resort: the whole level at once (only ever reached when the level
  // is no set of rooms, which the generator never draws).
  const whole = solvePush(lv, STATES_PER_ROOM);
  return whole !== null ? whole.moves : '';
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  let levelText: string;
  try {
    if (!statSync(join(INPUT_DIR, id)).isDirectory()) continue;
    levelText = readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8');
  } catch {
    continue;
  }
  const moves = solveLevel(levelText);
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
