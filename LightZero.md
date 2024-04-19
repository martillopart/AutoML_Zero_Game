# LightZero harness

These are the games supported by the LightZero repo

In short, we can easily adapt most of these games to our game. For instance, Atari, CartPole, LunarLander and such.

### This would be how

- To play our linear regression game, the player can move the 4 arrows and press the space bar.
- ➡️ move on to the next field
  - when on the last field, move to the next line
  - when on last field of the last line, submit program
- ⬆️ move to the next type of variable / operation / number / value
- ⬇️ change to the next increment for scalar:  { -10e1, -10e0, -10e-1, -10e-2, -10e-3, 10e-3, 10e-2, 10e-1, 10e0, 10e1 } – i.e. { -10, -1, -0.1,… 10 }
- ⬅️ reset token
  - press twice to delete line and move to the last field of the previous line
- <key>Space</key> add a new statement after this one
- Each line can only have up to a certain amount of characters.

### Operations

- There are 5 valid operations
- Opcode 7 is treated specially and denotes a section label; this opcode cannot be entered, and cannot be added / deleted / modified
- Opcode 6 is reserved; this opcode cannot be entered, and does not occur in the game

![image](https://github.com/martillopart/AutoML_Zero_Game/assets/51028/9ebb313b-4704-45ab-9d16-e8f0a9b6acb3)



### Notes

1. cursor starts on the first section label
2. labels are special statements that cannot be deleted, added, or modified
3. Newly added statement is the assignment s0 = 0.000
4. input generally wraps around, i.e. pressing the UP key on index 10 goes back to index zero, pressing DOWN changes the increment from 10 to -10, etc.
   1. the exception is moving forward through the program, which on the last token of the last line submits the program
5. when the cursor is on the section label, valid actions are move to the next line, and insert line
6. constants are rounded and clipped during encoding to 8-bit IEEE-like float, but the game interacts with them as unbounded decimals with precision 0.001
7. program can grow longer than the line limit, but submitting an overlong program results in a loss
   1. game is lost immediately once no combination of valid moves can result in a program that is within the line limit, i.e. when the number of lines behind the cursor has reached MAX_LINES + 1
8. game is lost if modifier input (UP key) is pressed more than one less than the number of times that can fully cycle once through the possible values for the given token position
   1. 4 for the opcode
   2. 9 for the variable indices 0..10
   3. 35 for the constant – this is likely a little generous; 25 or even 20 would suffice with greater error; see [https://chat.openai.com/share/b4dcad6a-b8a5-4e69-ad59-3aa5654b21e0](https://chat.openai.com/share/b4dcad6a-b8a5-4e69-ad59-3aa5654b21e0 "smartCard-inline")  for analysis

### Scoring

- the score is computed in a blackbox fashion by the C++ program
  - maximum score is 1
- all successful programs are recorded, and only later they are ranked according to latency (latency is not part of the score)

#### Penalties

- every line (small penalty TBC)
- submitted program too long (over the line limit; zero evaluation and small penalty)

# The one-player game would work as follows:

<strike>The player has to get to a fitness of 1 using the least amount of moves possible and the least amount of time to train. There will be a scoreboard with the two variables from the previous game. Scoring better means winning, scoring equal or worse means losing. For now we can just use moves and don't bother about the time.</strike>

The action space is limited enough to resemble one of the already implemented games so there will be no problem with that.

_And I think the time it will take one of these algorithms will not be that far from an evolutionary algorithm_
