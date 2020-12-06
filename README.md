# Digitizing chess moves from video

The goal of this repo is to extract and digitize a chess match from images or a video source. Images can be taken from any perspective of the board as long as all pieces are still partly visible. The digitization makes it possible to play against a chessbot or someone online on a physical chessboard.

The digitization is split into 3 parts. First the 4 corners of the chessboard are detected using a neural net performing keypoint detection. The board is then split into 64 images representing each checkboard cell. These images are classified with another neural net.

In the last step the current digitized board will be compared to previous boards to extract a chess move.

The moves are tracked to create the digitized chess match.

Processing of a frame takes roughly 0.8 seconds on a Laptop with i7-8550u and an MX150. Currently the detection and classification is only trained on this specific chessboard.
## Process  

<div style="text-align:center">
<img src="docs/process.png" width="500">
</div>

## Getting Started
### Prerequisites

To install the required packages listed in the requirements.txt:

- installation with pip:
    `pip install -r requirements.txt`

- installaion with conda:
    `conda install --file requirements.txt`

**It is important that the Tensorflow version is 2.2**

For a live analysis of the current chessboard place the stockfish binaries of your operating system here: `engine/stockfish`
### Run

To run the digitization and live analysis run:

`python3 detectionScript.py`