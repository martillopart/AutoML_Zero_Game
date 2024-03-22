from enter_alg import enter_alg
from evaluator import evaluate

def Round():
    player1_score = 0.0
    player2_score = 0.0
    while player1_score == player2_score:
        player1_score = 0.0
        player2_score = 0.0

        print ("Player 1 start entering algorithm:")
        alg = enter_alg()
        print ("Player 1 algorithm is:")
        print(alg)
        player1_score = evaluate(alg)

        print ("\n\nPlayer 2 start entering algorithm:")
        alg = enter_alg()
        print ("Player 2 algorithm is:")
        print(alg)
        player2_score = evaluate(alg)

        if player1_score > player2_score:
            print(f"Player 1 beat Player 2 with score {player1_score}:{player2_score}")
        elif player1_score < player2_score:
            print(f"Player 2 beat Player 1 with score {player2_score}:{player1_score} \n The game can continue")
            player1_score = 0.0
            player2_score = 0.0


if __name__ == "__main__":
    Round()
