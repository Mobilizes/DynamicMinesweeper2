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

        sum_local_solution = 0
        groups = (
            groups + [[]]
        )  # Adds an additional group that contains the parameters for the expressions that only have constants, which is an empty group
        alleq_groups = []  # Contains every possible solution for each group
        for i, eqgroup in enumerate(eq_groups):
            f = sympy.lambdify(groups[i], eqgroup)
            local_possible_solutions = []
            itr = 0
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
                if sum_local_solution > 150:
                    break
                itr += 1
                if itr > 100000:
                    break
            sum_local_solution += len(local_possible_solutions)
            alleq_groups.append(local_possible_solutions)
            if i > 20:
                break
        print(len(alleq_groups))

        end_time = time.time()  # End the timer
        print(f"Time taken to create alleq_groups: {end_time - start_time} seconds")
        print(f"Total number of possible solutions: {sum_local_solution}")
        print(f"Alleq groups: {len(alleq_groups)}")
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
            if len(unbordered_sqrs) > 0 and num_possibilities > 0:
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
