# Sokoban Solver — push-optimal, room by room

[![make.fun · sokoban-solver-01-run-20261121](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/fckkrel/sokoban-solver-fckkrel/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/fckkrel/sokoban-solver-fckkrel/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)

A push-optimal Sokoban solver in a single `main.ts`, for make.fun's "Build the best Sokoban solver, 21 November 2026" bounty.

- A level is a 1-wide hallway with small rooms off it; a box pushed out of a room can never return, so each room is solved on its own and the optima add up.
- Rooms are found by cutting the hallway's dead squares out of the floor.
- Each room is solved by A* over pushes, with dead-square and frozen-2x2 pruning and a cheapest box-to-goal assignment heuristic — so the first solution found is push-optimal.
- One run answers all 5 instances in ~0.3 s (limit 60 s), deterministic: no clock, no randomness, work-bounded only.

`main.ts` reads `/input/<id>/level.txt` and writes `/output/<id>/solution.txt` (LURD moves). It runs on plain Node 22, standard library only.
