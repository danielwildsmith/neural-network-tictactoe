import numpy as np
from joblib import load

# Load the model (Ensure the path is correct)
mlp_regressor = load('mlp_regressor.joblib')

def initialize_game():
    """Initialize the game state."""
    board = ["-", "-", "-",
             "-", "-", "-",
             "-", "-", "-"]
    game_on = True
    # By default, human (X) starts. This can be changed based on player choice.
    current_player = "X"
    return board, game_on, current_player

def display_board(board):
    """Display the current state of the board."""
    print(board[0] + " | " + board[1] + " | " + board[2] + "      " + "1|2|3")
    print(board[3] + " | " + board[4] + " | " + board[5] + "      " + "4|5|6")
    print(board[6] + " | " + board[7] + " | " + board[8] + "      " + "7|8|9\n")

def choose_first_player():
    """Let the player decide who starts the game."""
    choice = input("Do you want to go first? (Y/N): ").upper()
    while choice not in ["Y", "N"]:
        choice = input("Invalid choice. Please type Y for Yes or N for No: ").upper()
    return "X" if choice == "Y" else "O"

def player_position(board, current_player):
    """Manage player's and AI's moves."""
    if current_player == "X":  # Human
        position = input("Choose position from 1-9: ")
        valid = False
        while not valid:
            while position not in [str(num) for num in range(1, 10)]:
                position = input("Invalid input. Choose position from 1-9: ")
            position = int(position) - 1
            if board[position] == "-":
                valid = True
            else:
                print("Position already selected, choose another position!")
        board[position] = current_player
    else:  # AI
        move = get_model_move(board)
        board[move] = current_player
    display_board(board)

def get_model_move(board):
    """Generate a move for the AI based on the model prediction."""
    model_input = [1 if cell == "X" else -1 if cell == "O" else 0 for cell in board]
    model_input_np = np.array(model_input).reshape(1, -1)
    predictions = mlp_regressor.predict(model_input_np)[0]
    move = np.argmax(predictions)
    while board[move] != "-":
        predictions[move] = -np.inf  # Ensure not to select filled positions
        move = np.argmax(predictions)
    return move

def check_winner(board):
    global game_on
    # Rows, columns, and diagonals check
    win_conditions = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                      (0, 3, 6), (1, 4, 7), (2, 5, 8),
                      (0, 4, 8), (2, 4, 6)]
    for condition in win_conditions:
        if board[condition[0]] == board[condition[1]] == board[condition[2]] != "-":
            game_on = False
            return board[condition[0]]
    if "-" not in board:
        game_on = False
        return "Tie"
    return None

def flip_player(current_player):
    """Switch turns between X and O."""
    return "O" if current_player == "X" else "X"

def play_again():
    """Ask the player if they want to play another game."""
    response = input("Play again? (Y/N): ").upper()
    while response not in ["Y", "N"]:
        response = input("Invalid input. Please type Y for Yes or N for No: ").upper()
    return response == "Y"

def play_game():
    """The main game loop with support for replaying the game."""
    play = True
    while play:
        board, game_on, current_player = initialize_game()
        current_player = choose_first_player()
        print("Welcome to Tic Tac Toe! You are player 'X'\n")
        display_board(board)

        while game_on:
            player_position(board, current_player)
            winner = check_winner(board)
            if winner:
                if winner == "Tie":
                    print("It's a tie!")
                else:
                    print(f"{'The human' if winner == 'X' else 'MLP model'} won!")
                break
            current_player = flip_player(current_player)

        play = play_again()
        if play:
            print("\nStarting a new game...\n")
        else:
            print("Thanks for playing!")

# Start the game
play_game()
