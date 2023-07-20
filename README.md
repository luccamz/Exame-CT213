# Exame-CT213

## Install Dependencies

Use `pip install -r requirements.txt`

## Usage

First choose values for `BOARD_SZ` and `SLIPPERY` in `utils.py`. These define the size of the board and whether the ice is slippery, respectively, in the frozen lake environment.

Then run `train_agent.py` to train your agent.

Finally run `evaluate.py` for the agent evaluation.  For evaluation to display the environment rendering, change `RENDER` to `True` in the `evaluate.py`, then it will run with less episodes and show the rendering of the execution of each simulation.

## Warning

Don't change either `BOARD_SZ` or `SLIPPERY` in between training and evaluation.
