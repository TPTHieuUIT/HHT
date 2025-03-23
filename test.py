import random
import sys

import chess
import chess.svg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import (QApplication, QLabel, QPushButton, QVBoxLayout,
                             QWidget)


# Enhanced position evaluation
def evaluate_position(board):
    if board.is_checkmate():
        return -10000 if board.turn else 10000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    # Material evaluation with piece-square tables
    material_score = 0
    position_score = 0
    
    # Piece-square tables for positional evaluation
    # Pawns: Encourage center control and advancement
    pawn_table = [
        0, 0, 0, 0, 0, 0, 0, 0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5, 5, 10, 25, 25, 10, 5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, -5, -10, 0, 0, -10, -5, 5,
        5, 10, 10, -20, -20, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0
    ]
    
    # Knights: Encourage central positions
    knight_table = [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50
    ]
    
    # Bishops: Encourage diagonals and center control
    bishop_table = [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 10, 10, 10, 10, 0, -10,
        -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 5, 0, 0, 0, 0, 5, -10,
        -20, -10, -10, -10, -10, -10, -10, -20
    ]
    
    # Rooks: Encourage open files and 7th rank
    rook_table = [
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, 10, 10, 10, 10, 5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        0, 0, 0, 5, 5, 0, 0, 0
    ]
    
    # Queens: Avoid early development
    queen_table = [
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5,
        -10, 5, 5, 5, 5, 5, 0, -10,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20
    ]
    
    # Kings: Encourage castling and king safety in early/mid-game
    king_table_middlegame = [
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        20, 20, 0, 0, 0, 0, 20, 20,
        20, 30, 10, 0, 0, 10, 30, 20
    ]
    
    # Kings: Encourage activity in endgame
    king_table_endgame = [
        -50, -40, -30, -20, -20, -30, -40, -50,
        -30, -20, -10, 0, 0, -10, -20, -30,
        -30, -10, 20, 30, 30, 20, -10, -30,
        -30, -10, 30, 40, 40, 30, -10, -30,
        -30, -10, 30, 40, 40, 30, -10, -30,
        -30, -10, 20, 30, 30, 20, -10, -30,
        -30, -30, 0, 0, 0, 0, -30, -30,
        -50, -30, -30, -30, -30, -30, -30, -50
    ]
    
    # Determine game phase (roughly)
    pieces_count = len(board.pieces(chess.PAWN, chess.WHITE)) + len(board.pieces(chess.PAWN, chess.BLACK)) + \
                   len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK)) + \
                   len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK)) + \
                   len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK)) + \
                   len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
    
    is_endgame = pieces_count <= 12
    
    # Material and position evaluation
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0
    }
    
    for square, piece in board.piece_map().items():
        piece_type = piece.piece_type
        piece_value = piece_values[piece_type]
        
        # Get square position based on piece color
        sq_idx = square if piece.color == chess.BLACK else chess.square_mirror(square)
        
        # Material score
        if piece.color == chess.WHITE:
            material_score += piece_value
            
            # Position score
            if piece_type == chess.PAWN:
                position_score += pawn_table[sq_idx]
            elif piece_type == chess.KNIGHT:
                position_score += knight_table[sq_idx]
            elif piece_type == chess.BISHOP:
                position_score += bishop_table[sq_idx]
            elif piece_type == chess.ROOK:
                position_score += rook_table[sq_idx]
            elif piece_type == chess.QUEEN:
                position_score += queen_table[sq_idx]
            elif piece_type == chess.KING:
                if is_endgame:
                    position_score += king_table_endgame[sq_idx]
                else:
                    position_score += king_table_middlegame[sq_idx]
        else:  # BLACK
            material_score -= piece_value
            
            # Position score
            if piece_type == chess.PAWN:
                position_score -= pawn_table[sq_idx]
            elif piece_type == chess.KNIGHT:
                position_score -= knight_table[sq_idx]
            elif piece_type == chess.BISHOP:
                position_score -= bishop_table[sq_idx]
            elif piece_type == chess.ROOK:
                position_score -= rook_table[sq_idx]
            elif piece_type == chess.QUEEN:
                position_score -= queen_table[sq_idx]
            elif piece_type == chess.KING:
                if is_endgame:
                    position_score -= king_table_endgame[sq_idx]
                else:
                    position_score -= king_table_middlegame[sq_idx]
    
    # Additional strategic evaluations
    
    # Mobility: count legal moves (within reason)
    mobility = 0
    if board.turn == chess.WHITE:
        board.turn = chess.BLACK
        mobility -= min(len(list(board.legal_moves)), 25)  # Cap to avoid over-valuing
        board.turn = chess.WHITE
        mobility += min(len(list(board.legal_moves)), 25)
    else:
        board.turn = chess.WHITE
        mobility += min(len(list(board.legal_moves)), 25)
        board.turn = chess.BLACK
        mobility -= min(len(list(board.legal_moves)), 25)
    
    # Check and checkmate threats
    check_bonus = 0
    if board.is_check():
        check_bonus = -50 if board.turn == chess.WHITE else 50
    
    # Pawn structure
    pawn_structure = 0
    white_pawns = list(board.pieces(chess.PAWN, chess.WHITE))
    black_pawns = list(board.pieces(chess.PAWN, chess.BLACK))
    
    # Doubled pawns penalty
    for file in range(8):
        wp_count = sum(1 for sq in white_pawns if chess.square_file(sq) == file)
        bp_count = sum(1 for sq in black_pawns if chess.square_file(sq) == file)
        if wp_count > 1:
            pawn_structure -= 15 * (wp_count - 1)
        if bp_count > 1:
            pawn_structure += 15 * (bp_count - 1)
    
    # Center control bonus
    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    center_control = 0
    for sq in center_squares:
        if board.is_attacked_by(chess.WHITE, sq):
            center_control += 10
        if board.is_attacked_by(chess.BLACK, sq):
            center_control -= 10
    
    # Combine all evaluation factors with appropriate weights
    total_eval = (
        material_score
        + position_score * 0.3
        + mobility * 2
        + check_bonus
        + pawn_structure
        + center_control
    )
    
    return total_eval


# Simple transposition table to avoid repetitive moves
transposition_table = {}

# Improved minimax with alpha-beta pruning and move ordering
def minimax(board, depth, alpha, beta, maximizing_player, is_root=False):
    if depth == 0 or board.is_game_over():
        return evaluate_position(board)
    
    # Get board FEN for transposition table
    board_fen = board.fen()
    
    # Check transposition table for previously evaluated positions
    if not is_root and board_fen in transposition_table and transposition_table[board_fen][0] >= depth:
        return transposition_table[board_fen][1]
    
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    
    # Move ordering to improve alpha-beta pruning efficiency
    ordered_moves = []
    for move in legal_moves:
        # Score captures higher
        if board.is_capture(move):
            # MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)
            victim_value = 0
            victim_piece = board.piece_at(move.to_square)
            if victim_piece:
                piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                               chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
                victim_value = piece_values.get(victim_piece.piece_type, 0)
            
            ordered_moves.append((move, 10 + victim_value))
        # Score check moves higher
        elif board.gives_check(move):
            ordered_moves.append((move, 9))
        # Score promotions higher
        elif move.promotion:
            ordered_moves.append((move, 8))
        # Score moves to the center higher
        elif move.to_square in [chess.E4, chess.D4, chess.E5, chess.D5]:
            ordered_moves.append((move, 3))
        else:
            ordered_moves.append((move, 0))
    
    # Sort moves based on heuristic score (descending)
    ordered_moves.sort(key=lambda x: x[1], reverse=True)
    moves = [move for move, _ in ordered_moves]
    
    best_move = moves[0] if moves else None  # Default to first move in case nothing better is found
    
    # Check for repetitive positions - penalize them to avoid repetitions
    # This helps prevent the AI from suggesting the same moves repeatedly
    def is_repetitive(move):
        board.push(move)
        is_repeat = board.is_repetition(2)
        board.pop()
        return is_repeat
    
    # Filter out repetitive moves for the AI when at root level
    if is_root:
        non_repetitive_moves = [move for move in moves if not is_repetitive(move)]
        if non_repetitive_moves:  # If we have non-repetitive moves, prefer those
            moves = non_repetitive_moves
    
    if maximizing_player:
        max_eval = -float('inf')
        for move in moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        
        # Store result in transposition table
        if not is_root:
            transposition_table[board_fen] = (depth, max_eval)
        
        return best_move if is_root else max_eval
    else:
        min_eval = float('inf')
        for move in moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        
        # Store result in transposition table
        if not is_root:
            transposition_table[board_fen] = (depth, min_eval)
        
        return best_move if is_root else min_eval


# Iterative deepening for time-constrained search
def get_best_move(board, max_depth=3, avoid_repetition=True):
    best_move = None
    
    # Clear transposition table before each search to avoid outdated evaluations
    global transposition_table
    transposition_table = {}
    
    # Start with depth 1 and increase
    for depth in range(1, max_depth + 1):
        move = minimax(board, depth, -float('inf'), float('inf'), True, is_root=True)
        if move:
            best_move = move
    
    # If no good move found or the move would lead to a repetition, make a random legal move
    if not best_move or best_move not in board.legal_moves:
        legal_moves = list(board.legal_moves)
        if legal_moves:
            best_move = random.choice(legal_moves)
    elif avoid_repetition:
        # Check if best move leads to a repetition
        board.push(best_move)
        is_repeat = board.is_repetition(2)
        board.pop()
        
        # If it's a repetition, try to find another reasonable move
        if is_repeat:
            legal_moves = list(board.legal_moves)
            if len(legal_moves) > 1:  # If we have other options
                legal_moves.remove(best_move)
                # Just take a random move to break repetition
                best_move = random.choice(legal_moves)
    
    return best_move


class ChessGame(QWidget):
    def __init__(self):
        super().__init__()
        self.board = chess.Board()
        self.mode_selected = False
        self.player_vs_ai = False
        self.ai_suggestion = False
        self.selected_square = None
        self.suggested_move = None
        self.ai_difficulty = 3  # Default AI depth
        self.game_over = False  # Flag to track if the game is over
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.svg_widget = QSvgWidget()
        layout.addWidget(self.svg_widget)
        self.update_board()

        self.label = QLabel("Choose Mode (Cannot be changed during match):")
        layout.addWidget(self.label)

        self.p2p_button = QPushButton("Player vs Player")
        self.p2p_button.clicked.connect(lambda: self.set_mode(False))
        layout.addWidget(self.p2p_button)

        self.p2e_button = QPushButton("Player vs AI")
        self.p2e_button.clicked.connect(lambda: self.set_mode(True))
        layout.addWidget(self.p2e_button)

        self.ai_suggest_button = QPushButton("Toggle AI Suggest Move")
        self.ai_suggest_button.clicked.connect(self.toggle_ai_suggest)
        layout.addWidget(self.ai_suggest_button)
        
        # Add difficulty selection
        self.easy_button = QPushButton("Easy AI (Depth 2)")
        self.easy_button.clicked.connect(lambda: self.set_difficulty(2))
        layout.addWidget(self.easy_button)
        
        self.medium_button = QPushButton("Medium AI (Depth 3)")
        self.medium_button.clicked.connect(lambda: self.set_difficulty(3))
        layout.addWidget(self.medium_button)
        
        self.hard_button = QPushButton("Hard AI (Depth 4)")
        self.hard_button.clicked.connect(lambda: self.set_difficulty(4))
        layout.addWidget(self.hard_button)
        
        # Add a reset button
        self.reset_button = QPushButton("Reset Game")
        self.reset_button.clicked.connect(self.reset_game)
        layout.addWidget(self.reset_button)

        self.log_label = QLabel("Move Log:")
        layout.addWidget(self.log_label)

        self.setLayout(layout)
        self.setWindowTitle("Chess with AI")
        self.resize(600, 700)  # Increase the window size to accommodate the UI
        self.svg_widget.setFixedSize(400, 400)
        self.show()
    
    def reset_game(self):
        self.board = chess.Board()
        self.selected_square = None
        self.suggested_move = None
        self.game_over = False
        self.log_label.setText("Game reset!")
        self.update_board()

    def set_mode(self, is_ai):
        if not self.mode_selected:
            self.player_vs_ai = is_ai
            self.mode_selected = True
            self.p2p_button.setDisabled(True)
            self.p2e_button.setDisabled(True)
            self.log_label.setText("Mode selected. Game started!")
    
    def set_difficulty(self, depth):
        self.ai_difficulty = depth
        self.log_label.setText(f"AI difficulty set to: {depth}")

    def toggle_ai_suggest(self):
        self.ai_suggestion = not self.ai_suggestion
        status = "enabled" if self.ai_suggestion else "disabled"
        self.log_label.setText(f"AI move suggestion {status}")
        self.update_suggested_move()
        self.update_board()

    def update_suggested_move(self):
        if self.ai_suggestion and not self.game_over:
            try:
                self.suggested_move = get_best_move(self.board, self.ai_difficulty, avoid_repetition=True)
            except Exception as e:
                print(f"Error generating suggestion: {e}")
                self.suggested_move = None
        else:
            self.suggested_move = None

    def check_game_status(self):
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn else "White"
            self.log_label.setText(f"Game Over: {winner} wins by checkmate!")
            self.game_over = True
            return True
        elif self.board.is_stalemate():
            self.log_label.setText("Game Over: Stalemate! It's a draw!")
            self.game_over = True
            return True
        elif self.board.is_insufficient_material():
            self.log_label.setText("Game Over: Draw due to insufficient material!")
            self.game_over = True
            return True
        elif self.board.is_seventyfive_moves():
            self.log_label.setText("Game Over: Draw due to 75-move rule!")
            self.game_over = True
            return True
        elif self.board.is_fivefold_repetition():
            self.log_label.setText("Game Over: Draw due to fivefold repetition!")
            self.game_over = True
            return True
        return False

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() != Qt.LeftButton or not self.mode_selected or self.game_over:
            return

        log_text = ""
        
        # Use the fixed size of the SVG widget (400x400)
        board_size = 400
        square_size = board_size / 8  # 50 pixels per square

        # Get click coordinates relative to the SVG widget
        pos = self.svg_widget.mapFromGlobal(event.globalPos())
        x = pos.x()
        y = pos.y()

        print(f"Click coordinates: ({x}, {y})")

        # Check if click is within the board boundaries
        if 0 <= x < board_size and 0 <= y < board_size:
            # Calculate file (column) and rank (row)
            # File goes from 0 (a) to 7 (h), left to right
            file = int(x // square_size)
            # Rank goes from 7 (top) to 0 (bottom)
            rank = 7 - int(y // square_size)

            square = chess.square(file, rank)
            print(f"Detected square: {chess.square_name(square)} (file={file}, rank={rank})")

            if self.selected_square is None:
                # Select piece if one exists at the clicked square
                piece = self.board.piece_at(square)
                if piece and piece.color == self.board.turn:  # Only select pieces of the current player's color
                    self.selected_square = square
                    print(f"Selected: {chess.square_name(square)}")
            else:
                # Try to make a move
                move = chess.Move(self.selected_square, square)
                
                # Handle pawn promotion
                piece = self.board.piece_at(self.selected_square)
                if piece and piece.piece_type == chess.PAWN:
                    if (piece.color == chess.WHITE and rank == 7) or (piece.color == chess.BLACK and rank == 0):
                        move = chess.Move(self.selected_square, square, promotion=chess.QUEEN)

                if move in self.board.legal_moves:
                    piece = self.board.piece_at(self.selected_square)
                    self.board.push(move)
                    
                    # Log the move
                    move_log = f"Moved {piece.symbol().upper()} from {chess.square_name(self.selected_square)} to {chess.square_name(square)}"
                    if move.promotion:
                        move_log += f" (promoted to {chess.piece_symbol(move.promotion).upper()})"
                    log_text = move_log

                    self.selected_square = None
                    self.update_board()
                    
                    # Check game status after player move
                    if self.check_game_status():
                        self.update_board()
                        return

                    # AI move if in Player vs AI mode
                    if self.player_vs_ai and not self.board.turn and not self.game_over:
                        try:
                            ai_move = get_best_move(self.board, self.ai_difficulty, avoid_repetition=True)
                            if ai_move and ai_move in self.board.legal_moves:  # Safety check
                                from_piece = self.board.piece_at(ai_move.from_square)
                                move_desc = f"AI moved {from_piece.symbol().upper()} from {chess.square_name(ai_move.from_square)} to {chess.square_name(ai_move.to_square)}"
                                
                                self.board.push(ai_move)
                                
                                if ai_move.promotion:
                                    move_desc += f" (promoted to {chess.piece_symbol(ai_move.promotion).upper()})"
                                
                                log_text += f"\n{move_desc}"
                        except Exception as e:
                            print(f"Error during AI move: {e}")
                            log_text += "\nError occurred during AI move"
                        
                        self.update_board()
                        
                        # Check game status after AI move
                        if self.check_game_status():
                            self.update_board()
                            return

                    # Update AI suggestion if enabled and game is not over
                    if self.ai_suggestion and not self.game_over:
                        self.update_suggested_move()

                    # Position evaluation
                    if not self.game_over:
                        try:
                            eval_score = evaluate_position(self.board) / 100.0
                            eval_text = f"{eval_score:.2f}" if eval_score >= 0 else f"{eval_score:.2f}"
                            log_text += f"\nPosition evaluation: {eval_text} pawns"
                        except Exception as e:
                            print(f"Error during evaluation: {e}")
                    
                    self.log_label.setText(f"Move Log:\n{log_text}")
                else:
                    # If the move is not legal, deselect the square
                    self.selected_square = None

            self.update_board()
        else:
            print("Click outside board boundaries")

    def update_board(self):
        # Create the board SVG
        try:
            arrows = []
            if self.suggested_move is not None and not self.game_over:
                arrows = [(self.suggested_move.from_square, self.suggested_move.to_square)]
            
            last_move = None
            if not self.game_over and self.board.move_stack:
                last_move = self.board.peek()
            
            board_svg = chess.svg.board(
                self.board,
                squares=[self.selected_square] if self.selected_square else [],
                arrows=arrows,
                size=400,
                lastmove=last_move,
            )
            self.svg_widget.load(bytearray(board_svg, encoding='utf-8'))
        except Exception as e:
            print(f"Error updating board: {e}")
            # Fallback to rendering without last move
            try:
                board_svg = chess.svg.board(
                    self.board,
                    squares=[self.selected_square] if self.selected_square else [],
                    arrows=arrows if 'arrows' in locals() else [],
                    size=400
                )
                self.svg_widget.load(bytearray(board_svg, encoding='utf-8'))
            except Exception as e2:
                print(f"Critical error rendering board: {e2}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    game = ChessGame()
    sys.exit(app.exec_())