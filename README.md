# Real chessboard to FEN conversion.

The goal of this repo is to extract the board configuration from a chessboard image. Image can be from any perspective as long as all pieces are still visibel. Two neural nets are doing the conversion. One is detecting the board in the image while the other classifies each cell. Currently the nets are only trained on this specific chessboard.


#### Example
1. Input Image
    
    <img src="docs/orig.jpg" width="400">

2. Detection of board corners with ML pose estimation

    <img src="docs/orig_corners.png" width="400">

3. Classification of chess figures
    
    <img src="docs/board_img.png" width="300">
