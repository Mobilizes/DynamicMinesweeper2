from functions import *
from visual import *
import pygame as p
import sys
import time

# Function to display a settings menu and gather user input


def settings_menu():
    p.init()
    screen = p.display.set_mode((400, 300))
    p.display.set_caption("Minesweeper")

    font = p.font.Font(None, 32)
    input_boxes = [
        {"label": "Height", "value": "16", "rect": p.Rect(100, 50, 200, 32)},
        {"label": "Width", "value": "30", "rect": p.Rect(100, 100, 200, 32)},
        {"label": "Mines", "value": "99", "rect": p.Rect(100, 150, 200, 32)},
    ]

    current_box = 0
    active = [False] * len(input_boxes)
    active[current_box] = True

    clock = p.time.Clock()
    while True:
        screen.fill((200, 200, 200))  # Background color

        # Event handling
        for event in p.event.get():
            if event.type == p.QUIT:
                p.quit()
                sys.exit()
            elif event.type == p.KEYDOWN:
                if event.key == p.K_RETURN:  # Move to the next input
                    active[current_box] = False
                    current_box = (current_box + 1) % len(input_boxes)
                    active[current_box] = True
                elif event.key == p.K_BACKSPACE:
                    input_boxes[current_box]["value"] = input_boxes[current_box]["value"][:-1]
                else:
                    input_boxes[current_box]["value"] += event.unicode
            elif event.type == p.MOUSEBUTTONDOWN:
                for i, box in enumerate(input_boxes):
                    if box["rect"].collidepoint(event.pos):
                        active = [False] * len(input_boxes)
                        active[i] = True
                        current_box = i

        # Draw input boxes and labels
        for i, box in enumerate(input_boxes):
            color = (255, 255, 255) if active[i] else (180, 180, 180)
            p.draw.rect(screen, color, box["rect"], 0)
            txt_surface = font.render(box["value"], True, (0, 0, 0))
            screen.blit(txt_surface, (box["rect"].x + 5, box["rect"].y + 5))
            label = font.render(box["label"], True, (0, 0, 0))
            screen.blit(label, (20, box["rect"].y + 5))

        # Start button
        start_button = p.Rect(150, 220, 100, 40)
        p.draw.rect(screen, (100, 200, 100), start_button)
        start_text = font.render("Start", True, (255, 255, 255))
        screen.blit(start_text, (170, 225))

        # Error message for invalid inputs
        error_message = ""

        # Check for start button click
        if event.type == p.MOUSEBUTTONDOWN and start_button.collidepoint(event.pos):
            try:
                height = int(input_boxes[0]["value"])
                width = int(input_boxes[1]["value"])
                mines = int(input_boxes[2]["value"])

                # Validate height, width, and mines
                min_bombs = min(height, width) * 2
                max_bombs = height * width - 1

                if not (1 <= height <= 100 and 1 <= width <= 100):
                    error_message = "Height/Width must be between 1 and 100!"
                elif not (min_bombs <= mines <= max_bombs):
                    error_message = f"Mines must be between {min_bombs} and {max_bombs}!"
                else:
                    return height, width, mines  # Valid input, start the game
            except ValueError:
                error_message = "Please enter valid numbers!"

        # Display error message if invalid
        if error_message:
            error_surface = font.render(error_message, True, (255, 0, 0))
            screen.blit(error_surface, (50, 260))

        p.display.flip()
        clock.tick(30)


# Main program
if __name__ == "__main__":
    p.init()

    while True:  # Game loop to return to settings after losing
        height, width, num_mines = settings_menu()  # Show settings menu

        # Initialize game window after settings
        SQ_SIZE = 40
        num_flags = 0
        gameBoard = genBoard(height, width, num_mines)
        knownBoard = genKnownBoard(height, width)
        probsBoard = None
        seen = []
        lost = False
        display = False
        interval = 10
        click = 0

        screen = p.display.set_mode((width * SQ_SIZE, height * SQ_SIZE))
        clock = p.time.Clock()

        # Main game loop
        while not lost:
            drawBoard(screen, knownBoard, gameBoard,
                      probsBoard, SQ_SIZE, lost, display)
            clock.tick(MAX_FPS)
            p.display.flip()

            for e in p.event.get():
                if e.type == p.QUIT:
                    p.quit()
                    sys.exit()

                if e.type == p.KEYDOWN and e.key == p.K_c:
                    print("Auto-clearing the board using probabilities...")
                    knownBoard, gameBoard, seen = autoclear(knownBoard, gameBoard, seen,
                              num_mines, interval, click, screen, SQ_SIZE)
                    display = True  # Ensure probabilities are updated

                if e.type == p.MOUSEBUTTONDOWN and not lost:
                    location = p.mouse.get_pos()
                    col = location[0] // SQ_SIZE
                    row = location[1] // SQ_SIZE

                    if e.button == 1:  # Left click
                        if gameBoard[row][col] == 1:  # Mine clicked
                            print("It was a mine! Game Over!")
                            drawBoard(screen, knownBoard, gameBoard,
                                      probsBoard, SQ_SIZE, True, display)
                            p.display.flip()
                            time.sleep(2)
                            lost = True  # Exit the game loop
                        else:  # Safe square
                            knownBoard[row][col] = squareNum(
                                (row, col), gameBoard)
                            checkopencluster(knownBoard, gameBoard, seen)
                            click += 1

                    elif e.button == 3:  # Right click for flags
                        if knownBoard[row][col] is None:
                            knownBoard[row][col] = "🚩"
                            num_flags += 1
                        elif knownBoard[row][col] == "🚩":
                            knownBoard[row][col] = None
                            num_flags -= 1

                    probsBoard = calcprobs(knownBoard, num_mines - num_flags)
                    display = True

            if click == interval:
                num_flags = 0
                knownBoard, gameBoard, seen = randomizeBoard(
                    knownBoard, gameBoard, seen, num_mines)
                probsBoard = calcprobs(knownBoard, num_mines)
                click = 0
