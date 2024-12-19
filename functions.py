from random import choice, shuffle
from copy import deepcopy
import sympy
import itertools
from math import comb
from operator import mul
from functools import reduce
import pygame as p
from visual import *
from itertools import product
import time

# Generates a random mine board for the game
def genBoard(h, w, nb):
    board = []
    for i in range(h):
        board.append([0] * w)
    possibleCells = []
    for y in range(h):
        for x in range(w):
            possibleCells.append((y, x))
    for i in range(nb):
        c = choice(possibleCells)
        board[c[0]][c[1]] = 1
        possibleCells.remove(c)
    return board


# Generates the initial state of what the player can see of the board
def genKnownBoard(h, w):
    board = []
    for i in range(h):
        board.append([None] * w)
    return board


# Checks whether some square is inside or not the board
def inside(coords, h, w):
    if 0 <= coords[0] < h and 0 <= coords[1] < w:
        return True
    else:
        return False


# Returns all the surrounding squares of a given square
def surrounds(coords, board):
    y, x = coords[0], coords[1]
    possibilities = [
        (y + 1, x),
        (y + 1, x + 1),
        (y, x + 1),
        (y - 1, x),
        (y - 1, x - 1),
        (y, x - 1),
        (y + 1, x - 1),
        (y - 1, x + 1),
    ]
    surr_sqrs = []
    for sqr in possibilities:
        if inside(sqr, len(board), len(board[0])):
            surr_sqrs.append(sqr)
    return surr_sqrs


# Returns the number of mines around a given square
def squareNum(coords, board):
    y, x = coords[0], coords[1]
    possibilities = [
        (y + 1, x),
        (y + 1, x + 1),
        (y, x + 1),
        (y - 1, x),
        (y - 1, x - 1),
        (y, x - 1),
        (y + 1, x - 1),
        (y - 1, x + 1),
    ]
    bombcount = 0
    for sqr in possibilities:
        if inside(sqr, len(board), len(board[0])):
            bombcount += board[sqr[0]][sqr[1]]
    return bombcount


def randomizeBoard(knownBoard, gameBoard, seen, num_mines):
    new_known_board = genKnownBoard(len(knownBoard), len(knownBoard[0]))
    new_game_board = genBoard(len(gameBoard), len(gameBoard[0]), num_mines)
    new_seen = []

    opened_cells = 0
    for row in knownBoard:
        for cell in row:
            if cell is not None:
                opened_cells += 1

    positions = [(i, j) for i in range(len(new_known_board))
                 for j in range(len(new_known_board[0])) if new_game_board[i][j] != 1]
    shuffle(positions)

    # Open the same number of cells in the new known board
    for i in range(opened_cells):
        if i >= len(positions):
            break
        x, y = positions[i]
        new_known_board[x][y] = squareNum((x, y), new_game_board)
        checkopencluster(new_known_board, new_game_board, new_seen)
        positions.remove((x, y))

    return new_known_board, new_game_board, new_seen


# In minesweeper, when the player clicks a square with number 0, all surrouding squares are cleared automatically
# This function does that.


def autoclear(knownBoard, gameBoard, seen, num_mines, interval, click, screen=None, SQ_SIZE=40):
    clock = p.time.Clock()
    FPS = 10

    def in_3x3_around_known(y, x):
        """Check if a square (y, x) is in a 3x3 region around any revealed square."""
        for ky, row in enumerate(knownBoard):
            for kx, cell in enumerate(row):
                if isinstance(cell, int) and abs(ky - y) <= 1 and abs(kx - x) <= 1:
                    return True
        return False

    def get_safest_square(probsBoard):
        """Find the square with the smallest probability in the 3x3 area around revealed cells."""
        min_prob = 1.1
        min_square = None
        for y, row in enumerate(probsBoard):
            for x, prob in enumerate(row):
                if (
                    knownBoard[y][x] is None
                    and prob is not None
                    and prob < min_prob
                    and in_3x3_around_known(y, x)
                ):
                    min_prob = prob
                    min_square = (y, x)
        return min_square, min_prob

    def overlay_probs(probsBoard, probsBoard2):
        """Overlay probabilities using calcprobs and calcprobs2, prioritizing bombs."""
        opened_squares = []
        for y, row in enumerate(probsBoard):
            for x, prob in enumerate(row):
                prob2 = probsBoard2[y][x]
                if prob == 1.0:  # Definitive bomb from calcprobs
                    knownBoard[y][x] = "🚩"
                elif prob2 == 1.0:  # Bomb from calcprobs2
                    knownBoard[y][x] = "🚩"
                elif prob == 0.0 or prob2 == 0.0:  # Safe square
                    if knownBoard[y][x] is None:
                        knownBoard[y][x] = squareNum((y, x), gameBoard)
                        # Track squares we opened
                        opened_squares.append((y, x))
        return opened_squares

    def find_unexplored_cluster():
        """Identify a cluster of unexplored squares."""
        visited = set()
        clusters = []

        def dfs(y, x, cluster):
            if (y, x) in visited or knownBoard[y][x] is not None:
                return
            visited.add((y, x))
            cluster.append((y, x))
            for ny, nx in surrounds((y, x), knownBoard):
                dfs(ny, nx, cluster)

        for y, row in enumerate(knownBoard):
            for x, cell in enumerate(row):
                if cell is None and (y, x) not in visited:
                    cluster = []
                    dfs(y, x, cluster)
                    if cluster:
                        clusters.append(cluster)

        return max(clusters, key=len) if clusters else []

    # Initialization
    probsBoard = calcprobs(knownBoard, num_mines -
                           sum(row.count("🚩") for row in knownBoard))
    backtrack = 0

    while True:
        # Overlay results and open squares based on probabilities
        probsBoard2 = calcprobs2(
            knownBoard, num_mines - sum(row.count("🚩") for row in knownBoard))
        opened_squares = overlay_probs(probsBoard, probsBoard2)

        # Expand clusters from opened squares
        for y, x in opened_squares:
            checkopencluster(knownBoard, gameBoard, seen)

        # Find the safest square
        min_square, min_prob = get_safest_square(probsBoard)

        if min_square:
            y, x = min_square
            print(
                f"Choosing square {min_square} with probability {min_prob:.2f}")

            if gameBoard[y][x] == 1:  # Backtrack condition
                print(f"Backtrack {backtrack} at {min_square}")
                backtrack += 1
                knownBoard[y][x] = "🚩"
            else:  # Safe square
                knownBoard[y][x] = squareNum((y, x), gameBoard)
                checkopencluster(knownBoard, gameBoard, seen)

                # Update probabilities
                probsBoard = calcprobs(
                    knownBoard, num_mines - sum(row.count("🚩") for row in knownBoard))

        else:
            print("No safe moves found. Exploring the largest unexplored cluster.")
            unexplored_cluster = find_unexplored_cluster()
            if unexplored_cluster:
                # Select the first square in the largest cluster
                y, x = unexplored_cluster[0]
                print(f"Exploring random square {y, x}")
                if gameBoard[y][x] == 1:  # Bomb hit
                    print(f"Backtrack {backtrack} at {y, x}")
                    backtrack += 1
                    knownBoard[y][x] = "🚩"
                else:
                    knownBoard[y][x] = squareNum((y, x), gameBoard)
                    checkopencluster(knownBoard, gameBoard, seen)
                probsBoard = calcprobs(
                    knownBoard, num_mines - sum(row.count("🚩") for row in knownBoard))
            else:
                print("No unexplored regions left. Exiting autoclear.")
                break

        click += 1
        if click == interval:
            knownBoard, gameBoard, seen = randomizeBoard(
                knownBoard, gameBoard, seen, num_mines)
            probsBoard = calcprobs(knownBoard, num_mines)
            click = 0

        # Update the display
        if screen:
            drawBoard(screen, knownBoard, gameBoard,
                      probsBoard, SQ_SIZE, False, True)
            p.display.flip()
            clock.tick(FPS)

    print(f"Total backtracks: {backtrack}")

    return knownBoard, gameBoard, seen


def checkopencluster(knownBoard, gameBoard, seen):
    count = None
    while count != 0:
        count = 0
        zerosqrs = []
        for y, r in enumerate(knownBoard):
            for x, c in enumerate(r):
                if c == 0 and (y, x) not in seen:
                    for sqr in surrounds((y, x), gameBoard):
                        if gameBoard[sqr[0]][sqr[1]] == 0:
                            sqrNum = squareNum(sqr, gameBoard)
                            if sqrNum == 0:
                                count += 1
                                zerosqrs.append((sqr[0], sqr[1]))
                            else:
                                knownBoard[sqr[0]][sqr[1]] = sqrNum
                    seen.append((y, x))
        for sqr in zerosqrs:
            knownBoard[sqr[0]][sqr[1]] = 0


# Generates all combinations of a list l of symbols with length n
def foo(l, n):
    yield from itertools.product(*([l] * n))


# Breaks down a parametric solution of a linear system into groups
# of parameters that affect each other
def gen_groups(solution, parameters):
    groups = []
    for i in range(len(parameters)):
        groups.append([])
    for i, par in enumerate(parameters):
        groups[i].append(par)
        for eq in solution:
            if eq.coeff(par) != 0:
                for par2 in parameters:
                    if eq.coeff(par2) != 0 and par2 != par and par2 not in groups[i]:
                        groups[i].append(par2)
    for par in parameters:
        hasit = []
        for i, group in enumerate(groups):
            if par in group and group not in hasit:
                hasit.append(group)
        if len(hasit) > 1:
            newgroup = []
            for group in hasit:
                groups.remove(group)
                for par2 in group:
                    if par2 not in newgroup:
                        newgroup.append(par2)
            groups.append(newgroup)
    return groups


# The function that returns a board with the probability of each border square having a mine
def calcprobs(board, rem_mines):
    test = 1
    if (test):
        newboard = genKnownBoard(len(board), len(board[0]))
        test = 0
    else:
        newboard = calcprobs2(len(board), len(board[0]))
    prev = None
    # Basic rules (For example, "if the number of unknown surrounding squares is equal to the number of the square, then all of them have mines")
    while prev != newboard:
        prev = deepcopy(newboard)
        for y, row in enumerate(board):
            for x, cell in enumerate(row):
                if type(cell) == int:
                    if cell > 0:
                        sur_sqrs = surrounds((y, x), board)
                        none_sqrs = [
                            x
                            for x in sur_sqrs
                            if board[x[0]][x[1]] == None or board[x[0]][x[1]] == "🚩"
                        ]
                        sqrs100 = [
                            x
                            for x in none_sqrs
                            if newboard[x[0]][x[1]] == 1.0 or board[x[0]][x[1]] == "🚩"
                        ]
                        sqrs0 = [
                            x for x in none_sqrs if newboard[x[0]][x[1]] == 0.0]
                        if len(none_sqrs) == cell:
                            for sqr in none_sqrs:
                                newboard[sqr[0]][sqr[1]] = 1.0
                        elif len(sqrs100) == cell:
                            for sqr in [x for x in sur_sqrs if x not in sqrs100]:
                                newboard[sqr[0]][sqr[1]] = 0.0
                        elif len(none_sqrs) - len(sqrs0) == cell:
                            for sqr in [x for x in none_sqrs if x not in sqrs0]:
                                newboard[sqr[0]][sqr[1]] = 1.0

    # Uses groups and set ideas to determine which squares must have mines or not. Watch this video for better understanding: https://youtu.be/8j7bkNXNx4M
    border_sqrs = []
    for y, row in enumerate(board):
        for x, cell in enumerate(row):
            if type(cell) == int:
                if cell > 0 and None in [
                    board[sqr[0]][sqr[1]] for sqr in surrounds((y, x), board)
                ]:
                    border_sqrs.append((y, x))
    for sqr in border_sqrs:
        sur_borders = [
            x for x in surrounds((sqr[0], sqr[1]), board) if x in border_sqrs
        ]
        sur_unknown = [
            x
            for x in surrounds((sqr[0], sqr[1]), board)
            if board[x[0]][x[1]] == None and type(board[x[0]][x[1]]) != float
        ]
        sqr_val = board[sqr[0]][sqr[1]] - len(
            [
                x
                for x in surrounds((sqr[0], sqr[1]), board)
                if board[x[0]][x[1]] == "🚩"
                or (type(board[x[0]][x[1]]) == float and board[x[0]][x[1]] == 1.0)
            ]
        )
        for adj_sqr in sur_borders:
            adjsur_unknown = [
                x
                for x in surrounds((adj_sqr[0], adj_sqr[1]), board)
                if board[x[0]][x[1]] == None and type(board[x[0]][x[1]]) != float
            ]
            adjsqr_val = board[adj_sqr[0]][adj_sqr[1]] - len(
                [
                    x
                    for x in surrounds((adj_sqr[0], adj_sqr[1]), board)
                    if board[x[0]][x[1]] == "🚩"
                    or (type(board[x[0]][x[1]]) == float and board[x[0]][x[1]] == 1.0)
                ]
            )
            only_adjsur = [x for x in adjsur_unknown if x not in sur_unknown]
            only_sur = [x for x in sur_unknown if x not in adjsur_unknown]
            if adjsqr_val - sqr_val == len(only_adjsur):
                for sqr2 in only_adjsur:
                    newboard[sqr2[0]][sqr2[1]] = 1.0
                for sqr2 in only_sur:
                    newboard[sqr2[0]][sqr2[1]] = 0.0

    # This section generates, solves and determines all the possible solutions for the minesweeper linear system of equations

    # Gets all border squares which still are not known to be mines or not
    border_sqrs = []
    for y, row in enumerate(board):
        for x, cell in enumerate(row):
            if type(cell) == int:
                if cell > 0:
                    border_sqrs += [
                        x
                        for x in surrounds((y, x), board)
                        if board[x[0]][x[1]] == None
                        and type(newboard[x[0]][x[1]]) != float
                    ]
    newborders_sqrs = []
    for x in border_sqrs:
        if x not in newborders_sqrs:
            newborders_sqrs.append(x)
    border_sqrs = newborders_sqrs

    # The squares that are not a border square and are not a cleared square
    unbordered_sqrs = []
    for y, row in enumerate(board):
        for x, cell in enumerate(row):
            if (
                cell == None
                and ((y, x) not in border_sqrs)
                and type(newboard[y][x]) != float
            ):
                unbordered_sqrs.append((y, x))

    # Gets the equation for each number square
    equation_matrix = []
    for y, row in enumerate(board):
        for x, cell in enumerate(row):
            if type(cell) == int:
                if cell > 0:
                    equation = [0] * (len(border_sqrs) + 1)
                    for sqr in [
                        x
                        for x in surrounds((y, x), board)
                        if board[x[0]][x[1]] == None
                        and type(newboard[x[0]][x[1]]) != float
                    ]:
                        equation[border_sqrs.index(sqr)] = 1
                    equation[-1] = cell - len(
                        [
                            x
                            for x in surrounds((y, x), board)
                            if board[x[0]][x[1]] == "🚩" or newboard[x[0]][x[1]] == 1.0
                        ]
                    )
                    equation_matrix.append(equation)

    # Proceeds with solving the linear system if there are unknown squares probabilities at all
    if len(border_sqrs) > 0:
        # Names each border square with a1, a2, ...
        symbolstr = ""
        for i in range(len(border_sqrs)):
            symbolstr += f"a{i}, "
        symbolstr = symbolstr[:-2]
        variables = sympy.symbols(symbolstr)

        # Solve the system
        linsolve_result = sympy.linsolve(
            sympy.Matrix(equation_matrix), variables)

        # Check if linsolve_result is empty
        if not linsolve_result:
            print("No solutions found for the linear system. Returning early.")
            return newboard  # Return the current probability board as-is

        solution = linsolve_result.args[0]  # Extract the first solution tuple

        # Determines what are the parameters of the solution
        parameters = []
        for i in range(len(border_sqrs)):
            if isinstance(solution.args[i], sympy.Basic) and solution.args[i] == variables[i]:
                parameters.append(variables[i])
        print("Number of parameters", len(parameters))

        # The terms of each expression in the solution which are constants (For example, if an expression is a1+a2-2, then -2 is  the constant)
        constants = solution.subs(list(zip(parameters, [0] * len(parameters))))

        print("Separates the solution in groups that are independent between each other")
        # Separates the solution in groups that are independent between each other
        groups = gen_groups(solution, parameters)

        print("Gets the equation that belong to each group")
        # Gets the equations that belong to each group. An additional group is created to put the expressions which ONLY have constants
        eq_groups = []
        for i in range(len(groups) + 1):
            eq_groups.append([])
        # This is the number of the group that each expression in the solution belongs
        eq_groups_num = []
        for i, eq in enumerate(solution):
            num = None
            for j, group in enumerate(groups):
                for par in group:
                    if eq.coeff(par) != 0:
                        eq_groups[j].append(eq)
                        num = j
                        break
                    elif eq == constants[i]:
                        num = len(groups)
                        eq_groups[num].append(eq)
                        break
                else:
                    continue
                break
            eq_groups_num.append(num)

        print("Filter out None values before sorting")
        # Filter out None values before sorting
        filtered_pairs = [(group, sqr) for group, sqr in zip(
            eq_groups_num, border_sqrs) if group is not None]
        filtered_solution = [sol for group, sol in zip(
            eq_groups_num, solution) if group is not None]

        print("Sort the filtered pairs and corresponding solution")
        # Sort the filtered pairs and corresponding solution
        border_sqrs = [x for _, x in sorted(
            filtered_pairs, key=lambda pair: pair[0])]
        solution = [sol for _, sol in sorted(
            zip(eq_groups_num, filtered_solution), key=lambda pair: pair[0])]

        # if len(parameters) > 22:
        #     for i, sol in enumerate(solution):
        #         if sol == 0:  # Definite safe square
        #             newboard[border_sqrs[i][0]][border_sqrs[i][1]] = 0.0
        #         elif sol == 1:  # Definite bomb square
        #             newboard[border_sqrs[i][0]][border_sqrs[i][1]] = 1.0
        #     return calcprobs2(board, rem_mines)

        print("Creating alleq_groups")
        start_time = time.time()  # Start the timer

        groups = (
            groups + [[]]
        )  # Adds an additional group that contains the parameters for the expressions that only have constants, which is an empty group
        alleq_groups = []  # Contains every possible solution for each group
        for i, eqgroup in enumerate(eq_groups):
            f = sympy.lambdify(groups[i], eqgroup)
            local_possible_solutions = []
            for possibility in product([0, 1], repeat=len(groups[i])):
                localsolution = f(*possibility)
                valid = True
                for x in localsolution:
                    # Each square either has or doesn't have a mine
                    if x != 1 and x != 0:
                        valid = False
                        break
                if valid:
                    local_possible_solutions.append(localsolution)
                if len(alleq_groups) > 1000000:
                    break
                if len(local_possible_solutions) > 100:
                    break
            print(len(local_possible_solutions))
            alleq_groups.append(local_possible_solutions)
            if i > 20:
                break
        print(len(alleq_groups))

        end_time = time.time()  # End the timer
        print(f"Time taken to create alleq_groups: {end_time - start_time} seconds")
        print(len(alleq_groups))

        # It doesn't continue if there are too many parameters, because it could take a lot of time
        # The number of maximum parameters can be changing depending on the speed of the computer
        if len(parameters) < 22:
            # The number of mines that already have been found by the previous two methods
            alreadyFoundMines = 0
            for y, row in enumerate(newboard):
                for x, cell in enumerate(row):
                    if cell == 1.0 and board[y][x] != "🚩":
                        alreadyFoundMines += 1

            # A dictionary to reuse some big values already calculated with comb()
            numberOfPossibilities = {}
            problist = [0] * len(
                border_sqrs
            )  # The probabilities of each respective border square having a mine
            unbordered_prob = (
                0  # The probability of each unbordered square having a mine
            )
            total = 0  # Total number of possible states of mines
            num_possibilities = (
                0  # counts the number of possible states of the bordered squares
            )

            # Generates every possible combination of all the possible solutions for each group
            for c in itertools.product(*[list(range(len(x))) for x in alleq_groups]):
                result = []
                for x in [alleq_groups[i][j] for i, j in enumerate(c)]:
                    result = (
                        result + x
                    )  # Because the border squares were sorted based on the group number, the group possible solutions can just be summed to the list

                val = sum(result)
                if (
                    val <= rem_mines - alreadyFoundMines
                ):  # The number of mines in the current possible solution can't be bigger than the number of remaining mines
                    if val not in numberOfPossibilities:
                        numberOfPossibilities[val] = float(comb(
                            len(unbordered_sqrs), rem_mines -
                            val - alreadyFoundMines
                        ))
                    total += numberOfPossibilities[val]
                    unbordered_prob += rem_mines - val - alreadyFoundMines
                    num_possibilities += 1
                    for i, x in enumerate(result):
                        problist[i] += x * numberOfPossibilities[val]

            # Normalize probabilities after solving
            if total > 0:
                # Normalize probabilities
                problist = [x / total for x in problist]
            else:
                problist = [0] * len(problist)  # No valid states
            if len(unbordered_sqrs) > 0:
                unbordered_prob /= num_possibilities * len(unbordered_sqrs)

            # Assign probabilities to newboard
            for i, sqr in enumerate(border_sqrs):
                newboard[sqr[0]][sqr[1]] = problist[i]
            for i, sqr in enumerate(unbordered_sqrs):
                newboard[sqr[0]][sqr[1]] = rem_mines / \
                    len(unbordered_sqrs)  # Uniform probability
        else:
            # In case there were too many parameters, at least some squares are definitely mines because the
            # Iterate over the solution and check if it matches 0 or 1
            # Terganti
            for i, sol in enumerate(solution):
                if sol == 0:  # Definite safe square
                    newboard[border_sqrs[i][0]][border_sqrs[i][1]] = 0.0
                elif sol == 1:  # Definite bomb square
                    newboard[border_sqrs[i][0]][border_sqrs[i][1]] = 1.0
            return calcprobs2(board, rem_mines)
    else:
        print("The linear system method wasn't needed")
    return newboard


def calcprobs2(board, rem_mines):
    # Initialize probability board with None
    newboard = genKnownBoard(len(board), len(board[0]))
    prev = None

    # Identify revealed cells to limit the scope of calculation
    revealed_cells = [
        (y, x) for y, row in enumerate(board) for x, cell in enumerate(row) if isinstance(cell, int)
    ]

    # Focus only on cells surrounding the revealed cells
    def get_surrounding_revealed():
        focus_cells = set()
        for y, x in revealed_cells:
            for ny, nx in surrounds((y, x), board):
                if board[ny][nx] is None:  # Only consider unknown cells
                    focus_cells.add((ny, nx))
        return list(focus_cells)

    while prev != newboard:
        prev = deepcopy(newboard)
        focus_cells = get_surrounding_revealed()

        for y, x in focus_cells:  # Iterate only over the cells of interest
            for ry, rx in surrounds((y, x), board):
                cell = board[ry][rx]
                if isinstance(cell, int) and cell > 0:
                    sur_sqrs = surrounds((ry, rx), board)
                    none_sqrs = [
                        s for s in sur_sqrs if board[s[0]][s[1]] is None or board[s[0]][s[1]] == "🚩"
                    ]
                    sqrs100 = [
                        s for s in none_sqrs if newboard[s[0]][s[1]] == 1.0 or board[s[0]][s[1]] == "🚩"
                    ]
                    sqrs0 = [s for s in none_sqrs if newboard[s[0]][s[1]] == 0.0]

                    # Apply basic rules for definite mines or safe squares
                    if len(none_sqrs) == cell:
                        for sqr in none_sqrs:
                            newboard[sqr[0]][sqr[1]] = 1.0
                    elif len(sqrs100) == cell:
                        for sqr in [s for s in sur_sqrs if s not in sqrs100]:
                            newboard[sqr[0]][sqr[1]] = 0.0
                    elif len(none_sqrs) - len(sqrs0) == cell:
                        for sqr in [s for s in none_sqrs if s not in sqrs0]:
                            newboard[sqr[0]][sqr[1]] = 1.0

                    # Calculate probabilities for uncertain squares
                    else:
                        # Mines yet to be placed
                        mines_left = cell - len(sqrs100)
                        if mines_left > 0 and len(none_sqrs) > 0:
                            prob = mines_left / len(none_sqrs)
                            for sqr in none_sqrs:
                                # Update only unset probabilities
                                if newboard[sqr[0]][sqr[1]] is None:
                                    newboard[sqr[0]][sqr[1]] = prob
                                else:
                                    # Average probability for overlapping regions
                                    newboard[sqr[0]][sqr[1]] = (
                                        newboard[sqr[0]][sqr[1]] + prob
                                    ) / 2

    # Normalize probabilities if needed
    total_prob = sum(
        newboard[y][x]
        for y, row in enumerate(newboard)
        for x, cell in enumerate(row)
        # Exclude definite bombs
        if isinstance(cell, float) and newboard[y][x] != 1.0
    )

    if (rem_mines > 0):
        if total_prob > rem_mines:  # Normalize probabilities to match remaining mines
            scale_factor = rem_mines / total_prob
            for y, row in enumerate(newboard):
                for x, cell in enumerate(row):
                    # Do not modify 1.0
                    if isinstance(cell, float) and newboard[y][x] != 1.0:
                        newboard[y][x] = round(
                            min(1.0, newboard[y][x] * scale_factor), 2)

    return newboard


from itertools import product
from functools import reduce
from math import factorial
from operator import mul

import numpy as np
from scipy.ndimage.measurements import label
from constraint import Problem, ExactSumConstraint, MaxSumConstraint


class Solver:
    def __init__(self, width, height, total_mines, stop_on_solution=True):
        """ Initialize the solver.
            :param width: The width of the minefield.
            :param height: The height of the minefield.
            :param total_mines: The total number of mines on the minefield, flagged or not.
            :param stop_on_solution: Whether to stop as soon as a square is found that can be opened with 100%
                                     certainty. This means that the solution may have np.nan in it for squares that
                                     weren't computed.
        """
        # Known mines are 1, known empty squares are 0, uncertain squares are np.nan.
        self._total_mines = total_mines
        self.known = np.full((height, width), np.nan)
        self._stop_on_solution = stop_on_solution

    def known_mine_count(self):
        """ Returns how many mines we know the location of. """
        return (self.known == 1).sum(dtype=int)

    def mines_left(self):
        """ Returns the number of mines that we don't know the location of yet. """
        return self._total_mines - self.known_mine_count()

    def solve(self, state):
        """ Compute the probability of there being a mine under a given square.
            :param state: A 2D nested list representing the minesweeper state. Values can be a number in the range
                          [0, 8], 'flag', '?' or np.nan for any unopened squares.
            :returns: An array giving the probability that each square contains a mine. If `stop_on_solution` is set, a
                      partially computed result may be returned with a number of squares being np.nan, as they
                      weren't computed yet. Squares that have already been opened will also be np.nan.
        """
        # Convert to an easier format to solve: only the numbers remain, the rest is np.nan.
        state = np.array([[state[y][x] if isinstance(state[y][x], int) else np.nan for x in range(len(state[0]))] for y in range(len(state))])

        # Are there any opened squares; is this the first move?
        if not np.isnan(state).all():
            # Expand the known state with new information from the passed state.
            self.known[~np.isnan(state)] = 0
            # Reduce the numbers and check if there are any trivial solutions, where we have a 0 with neighbors or a
            # square with the number N in it and N unflagged neighbors.
            prob, state = self._counting_step(state)
            # Stop early if the `stop_on_solution` flag is set and we've found a safe square to open.
            if self._stop_on_solution and ~np.isnan(prob).all() and 0 in prob:
                return prob
            # Compute the probabilities for the remaining, uncertain squares.
            prob = self._cp_step(state, prob)
            return prob
        else:
            # If no cells are opened yet, just give each cell the same probability.
            return np.full(state.shape, self._total_mines/state.size)

    def _counting_step(self, state):
        """ Find all trivially easy solutions. There are 2 cases we consider:
            - A square with a 0 in it and has unflagged and unopened neighbors means that we can open all neighbors.
            - 1 square with a number that matches the number of unflagged and unopened neighbors means that we can flag
              all those neigbors.

            :param state: The unreduced state of the minefield
            :returns result: An array with known mines marked with 1, squares safe to open with 0 and everything else
                             as np.nan.
            :returns reduced_state: The reduced state, where numbers indicate the number of neighboring mines that have
                                    *not* been found.
        """
        result = np.full(state.shape, np.nan)
        # This step can be done multiple times, as each time we have results, the numbers can be further reduced.
        new_results = True
        # Subtract all numbers by the amount of neighboring mines we've already found, simplifying the game.
        state = reduce_numbers(state, self.known == 1)
        # Calculate the unknown square, i.e. that are unopened and we've not previously found their value.
        unknown_squares = np.isnan(state) & np.isnan(self.known)
        while new_results:
            num_unknown_neighbors = count_neighbors(unknown_squares)
            ### First part: squares with the number N in it and N unflagged/unopened neighbors => all mines.
            # Calculate squares with the same amount of unflagged neighbors as neighboring mines (except if N==0).
            solutions = (state == num_unknown_neighbors) & (num_unknown_neighbors > 0)
            # Create a mask for all those squares that we now know are mines. The reduce makes a neighbor mask for each
            # solution and or's them together, making one big neighbors mask.
            known_mines = unknown_squares & reduce(np.logical_or,
                [neighbors_xy(x, y, state.shape) for y, x in zip(*solutions.nonzero())], np.zeros(state.shape, dtype=bool))
            # Update our known matrix with these new finding: 1 for mines.
            self.known[known_mines] = 1
            # Further reduce the numbers, since we found new mines.
            state = reduce_numbers(state, known_mines)
            # Update what is unknown by removing known flags from the `unknown_squares` mask.
            unknown_squares = unknown_squares & ~known_mines
            # The unknown neighbor count might've changed too, so recompute it.
            num_unknown_neighbors = count_neighbors(unknown_squares)

            ### Second part: squares with a 0 in and any unflagged/unopened neighbors => all safe.
            # Calculate the squares that have a 0 in them, but still have unknown neighbors.
            solutions = (state == 0) & (num_unknown_neighbors > 0)
            # Select only those squares that are unknown and we've found to be neighboring any of the found solutions.
            # The reduce makes a neighbor mask for each solution and or's them together, making one big neighbor mask.
            known_safe = unknown_squares & reduce(np.logical_or,
                [neighbors_xy(x, y, state.shape) for y, x in zip(*solutions.nonzero())], np.zeros(state.shape, dtype=bool))
            # Update our known matrix with these new finding: 0 for safe squares.
            self.known[known_safe] = 0
            # Update what is unknown.
            unknown_squares = unknown_squares & ~known_safe

            # Now update the result matrix for both steps, 0 for safe squares, 1 for mines.
            result[known_safe] = 0
            result[known_mines] = 1
            new_results = (known_safe | known_mines).any()
        return result, state

    def _cp_step(self, state, prob):
        """ The constraint programming step.

            This is one of the more complicated steps; it divides the boundary into
            components that don't influence each other first, then divides each of those into areas that are equally
            constrained and must therefore have the same probabilities. The combinations of the number of mines in those
            components is computed with constraint programming. Those solutions are then combined to count the number of
            models in which each area has the given number of mines, from which we can calculate the average expected
            number of mines per square in a component if it has M mines, i.e. per component we have a mapping of
            {num_mines: (num_models, avg_prob)}. This information is then passed on to the combining step to form the
            final probabilities.

            :param state: The reduced state.
            :param prob: The already computed probabilities.
            :returns: The exact probability for every unknown square.
        """
        components, num_components = self._components(state)
        c_counts = []   # List of model_count_by_m instances from inside the 'for c' loop.
        c_probs = []    # List of model_count_by_m instances from inside the 'for c' loop.
        m_known = self.known_mine_count()
        # Solve each component individually
        for c in range(1, num_components+1):
            areas, constraints = self._get_areas(state, components == c)
            # Create a CP problem to determine which combination of mines per area is possible.
            problem = Problem()
            # Add all variables, each one having a domain [0, num_squares].
            for v in areas.values():
                problem.addVariable(v, range(len(v)+1))
            # Now constrain how many mines areas can have combined.
            for constraint in constraints:
                problem.addConstraint(constraint, [v for k, v in areas.items() if constraint in k])
            # Add a constraint so that the maximum number of mines never exceeds the number of mines left.
            problem.addConstraint(MaxSumConstraint(self._total_mines - m_known), list(areas.values()))
            solutions = problem.getSolutions()
            model_count_by_m = {}       # {m: #models}
            model_prob_by_m = {}        # {m: prob of the average component model}
            # Now count the number of models that exist for each number of mines in that component.
            for solution in solutions:
                m = sum(solution.values())
                # Number of models that match this solution.
                model_count = self._count_models(solution)
                # Increase counter for the number of models that have m mines.
                model_count_by_m[m] = model_count_by_m.get(m, 0) + model_count
                # Calculate the probability of each square in the component having a mines.
                model_prob = np.zeros(prob.shape)
                for area, m_area in solution.items():
                    # The area has `m_area` mines in it, evenly distributed.
                    model_prob[tuple(zip(*area))] = m_area/len(area)
                # Sum up all the models, giving the expected number of mines of all models combined
                model_prob_by_m[m] = model_prob_by_m.get(m, np.zeros(prob.shape)) + model_count*model_prob
            # We've summed the probabilities of each solution, weighted by the number of models with those
            # probabilities, now divide out the total number of models to obtain the probability of each square of a
            # model with m mines having a mine.
            model_prob_by_m = {m: model_prob/model_count_by_m[m] for m, model_prob in model_prob_by_m.items()}
            c_probs.append(model_prob_by_m)
            c_counts.append(model_count_by_m)
        prob = self._combine_components(state, prob, c_probs, c_counts)
        return prob

    def _combine_components(self, state, prob, c_probs, c_counts):
        """ Combine the probabilities and model counts found in the CP step into one probability array.

            The combining is done by forming all combinations of mine counts for each component, without exceeding the
            total number of mines in the game, and a number proportional to the total number of models that exist per
            combination. The exact probability is then the individual probabilities weighed by the number of models,
            divided by the total number of models.

            :param state: The reduced state.
            :param prob: The already computed probabilities.
            :param c_probs: A list of probability mappings per component, each having the format {num_mines: prob}
            :param c_probs: A list of model count mappings per component, each having the format {num_mines: model count}
            :returns: The exact probability for every unknown square.
        """
        # Find the unconstrained squares, for which we need to calculate the weight of models with m mines.
        solution_mask = boundary(state) & np.isnan(self.known)
        unconstrained_squares = np.isnan(state) & ~solution_mask & np.isnan(self.known)
        n = unconstrained_squares.sum(dtype=int)
        # It's possible there aren't any components at all, e.g. when an area is cut off by mines, so just skip this.
        if c_probs:
            # Instead of calculating the total number of models, we calculate weights that are proportional to the
            # number of models of the combined components, i.e. w ∝ #models. This number is a lot smaller.
            min_mines = sum([min(d) for d in c_probs])
            max_mines = sum([max(d) for d in c_probs])
            mines_left = self.mines_left()
            weights = self._relative_weights(range(min_mines, min(max_mines, mines_left)+1), n)
            # Accumulate weights and probabilities inside the upcoming for loop, where combinations of components are
            # processed.
            total_weight = 0  # The weight of solutions combined.
            total_prob = np.zeros(prob.shape)  # The summed weighted probabilities.
            # Iterate over all combinations of the components.
            for c_ms in product(*[d.keys() for d in c_probs]):
                m = sum(c_ms)
                if self.mines_left() - n <= m <= mines_left:
                    # Combine the prob arrays for this component combination.
                    comb_prob = reduce(np.add, [c_probs[c][c_m] for c, c_m in enumerate(c_ms)])
                    comb_model_count = reduce(mul, [c_counts[c][c_m] for c, c_m in enumerate(c_ms)])
                    weight = weights[m] * comb_model_count
                    # Sum up the weights and the weighted probabilities.
                    total_weight += weight
                    total_prob += weight * comb_prob
            # Now normalize the probabilities by dividing out the total weight.
            total_prob /= total_weight
            # Add result to the prob array.
            prob[solution_mask] = total_prob[solution_mask]
        # If there are any unconstrained mines...
        if n > 0:
            m_known = self.known_mine_count()
            # The amount of remaining mines is distributed evenly over the unconstrained squares.
            prob[unconstrained_squares] = (self._total_mines - m_known - prob[~np.isnan(prob) & np.isnan(self.known)].sum()) / n
        # Remember the certain values.
        certain_mask = np.isnan(self.known) & ((prob == 0) | (prob == 1))
        self.known[certain_mask] = prob[certain_mask]
        return prob

    def _count_models(self, solution):
        """ Count how many models are possible for the solution of the component areas.
            :param solution: A solution to component areas, being a dictionary {area_key: number_or_mines}.
            :returns: How many ways the component's areas can be filled to match the solution.
        """
        # Multiply the number of combinations for each individual area to get the number of models.
        return reduce(mul, [self.combinations(len(area), m) for area, m in solution.items()])

    def _components(self, state):
        """ Find all connected components in the boolean array.

            In this case, a connected component is not quite like the typical mathematical concept, as components can be
            next to each other without influencing each other, so whereas they would be part of the same component in
            the typical sense, they are not in the minesweeper sense. Furthermore, a pair of traditional components that
            are not neighbors may still be connected in minesweeper, as they could have a number in between them that
            connects the two, causing information from one traditional component to affect another component.
        """
        # Get the numbers next to unknown borders.
        numbers_mask = dilate(np.isnan(state) & np.isnan(self.known)) & ~np.isnan(state)
        labeled, num_components = label(numbers_mask)
        # Get the section of the boundary that corresponds to the previously obtained numbers.
        number_boundary_masks = [neighbors(labeled == c) & np.isnan(self.known) & np.isnan(state) for c in range(1, num_components+1)]
        # Now merge components that overlap.
        i = 0   # (For once, a C-style for loop would be nice)
        while i < len(number_boundary_masks)-1:
            j = i + 1
            while j < len(number_boundary_masks):
                # If the components overlap...
                if (number_boundary_masks[i] & number_boundary_masks[j]).any():
                    # Merge the components.
                    number_boundary_masks[i] = number_boundary_masks[i] | number_boundary_masks[j]
                    del number_boundary_masks[j]
                    # The mask that was merged in may overlap with already checked masks, so loop with the same i.
                    i -= 1  # -1, so the outer while will iterate it back to the same i after we break.
                    break
                j += 1
            i += 1
        # Number each of the resulting boundaries, as with scipy's `label`.
        labeled = np.zeros(state.shape)
        num_components = len(number_boundary_masks)
        for c, mask in enumerate(number_boundary_masks, 1):
            labeled[mask] = c
        # Now connect components that have a number in between them that connect them with a constraint.
        i = 1
        while i <= num_components-1:
            j = i + 1
            while j <= num_components:
                # If there is a number connecting the two components...
                if not np.isnan(state[dilate(labeled == i) & dilate(labeled == j)]).all():
                    # Merge the components.
                    labeled[labeled == j] = i
                    labeled[labeled > j] -= 1
                    num_components -= 1
                    # The mask that was merged in may overlap with already checked masks, so loop with the same i.
                    i -= 1  # -1, so the outer while will iterate it back to the same i after we break.
                    break
                j += 1
            i += 1
        return labeled, num_components

    @staticmethod
    def _get_areas(state, mask):
        """ Split the masked area into regions, for which each square in that region is constrained by the same
            constraints.
            :returns mapping: A mapping of constraints to an n-tuple of squares it applies to. Each of these constraint
                              tuples uniquely defines an area.
            :returns constraints: A list of all constraints applicable in the component.
        """
        # Find all squares that contain numbers that are constraints for the given state.
        constraints_mask = neighbors(mask) & ~np.isnan(state)
        # Generate a list of all CP constraints corresponding to numbers.
        constraint_list = [ExactSumConstraint(int(num)) for num in state[constraints_mask]]
        # Now create an array where those constraints are placed in the corresponding squares.
        constraints = np.full(state.shape, None, dtype=object)
        constraints[constraints_mask] = constraint_list
        # Now create an array where a list of *applicable* constraints is stored for each variable in the boundary,
        # i.e. each square aggregates the constraints defined in its neighbors.
        applied_constraints = np.empty(state.shape, dtype=object)
        for y, x in zip(*mask.nonzero()):
            applied_constraints[y, x] = []
        for yi, xi in zip(*constraints_mask.nonzero()):
            constrained_mask = neighbors_xy(xi, yi, mask.shape) & mask
            for yj, xj in zip(*constrained_mask.nonzero()):
                applied_constraints[yj, xj].append(constraints[yi, xi])
        # Now group all squares that have the same constraints applied to them, each one being an area.
        mapping = {}
        for yi, xi in zip(*mask.nonzero()):
            k = tuple(applied_constraints[yi, xi])  # Convert to tuple, so we can use it as a hash key.
            if k not in mapping:
                mapping[k] = []
            mapping[k].append((yi, xi))
        # Turn the list of (x, y) tuples into a tuple, which allows them to be used as hash keys.
        mapping = {k: tuple(v) for k, v in mapping.items()}
        return mapping, constraint_list

    @staticmethod
    def combinations(n, m):
        """ Calculate the number of ways that m mines can be distributed in n squares. """
        return factorial(n)/(factorial(n-m)*factorial(m))

    def _relative_weights(self, ms_solution, n):
        """ Compute the relative weights of solutions with M mines and N squares left. These weights are proportional
            to the number of models that have the given amount of mines and squares left.

            The number of models with N squares and M mines left is C(N, M) = N!/((N-M)!M!). To understand this formula,
            realise that the numerator is the number of permutations of N squares. That number doesn't account for the
            permutations that have identical results. This is because two mines can be swapped without the result
            changing, the same goes for empty square. The denominator divides out these duplicates. (N-M)! divides out
            the number ways that empty squares can form duplicate results and M! divides out the the number of ways
            mines can form duplicate results. C(N, M) can be used to weigh solutions, but since C(N, M) can become very
            large, we can instead compute how much more a solution weighs through compared to a solution with a
            different C(N, M').

            We actually don't need to know or calculate C(N, M), we just need to know how to weigh solutions relative to
            each other. To find these relative weights we look at the following series of equalities:

            C(N, M+1) = N! / ((N-(M+1))! (M+1)!)
                      = N! / (((N-M)!/(N-M+1)) M!(M+1))
                      = N! / ((N-M)! M!) * (N-M+1)/(M+1)
                      = C(N, M) * (N-M+1) / (M+1)
            Or alternatively: C(N, M) = C(N, M-1) * (N-M)/M
            Notice that there is however an edge case where this equation doesn't hold, where N=M+1, you'd have
            a division by zero within (N-M)!/(N-M+1).

            So, a solution with C(N, M) models weighs (N-M)/M times more than a solution with C(N, M-1) models, allowing
            us to inductively calculate relative weights.

            :param ms_solution: A list of the number of mines in the solution, from which we derive the number of mines
                                left outside of the solution, which is the M from the equations above..
            :param n: The number of empty squares left.
            :returns: The relative weights for each M, as a dictionary {M: weight}.
        """
        mines_left = self.mines_left()
        # Special case: N == 0, all solutions should have the same weight.
        if n == 0:
            return {m: 1 for m in ms_solution}
        ms = sorted(ms_solution, reverse=True)
        m = mines_left-ms[0]
        weight = 1
        weights = {}
        for m_next_solution in ms:
            m_next = mines_left - m_next_solution
            # Iteratively compute the weights, using the results computed above to update the weight.
            for m in range(m+1, m_next+1):
                # Special case: m == n, due to factorial in the derivation.
                if n == m:
                    weight *= 1/m
                else:
                    weight *= (n-m)/m
            weights[m_next_solution] = weight
        return weights
