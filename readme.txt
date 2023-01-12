README: InRoad Duckie Avoidance

- Operating System: Ubuntu 20.04
- Packages versions:
    - py-multihash: 0.2.0
    - Pillow: 8.3.2
    - pyglet: 1.5.11
    - testresources: should be installed, not in the requirements of gym-duckietown
    - numpy package: 1.23
    - PyTorch: 1.8.1+cu101

- Directory Organization: 
DuckieAvoidance/
├── data/
│   ├── images/
│   └── labels/
├── dataset/
│   ├── annotation/
│   └── frames/
├── images/
├── model/
│   ├── exp/
│   ├── exp2/
│   └── exp3/
├── yolov5/
├── annotationFormat.py
├── DuckieDetector.py
├── DuckieDetectorML.py
├── LaneDetector.py
├── main.py
├── readme.txt
├── Simulator.py
├── test.py
├── TestLaneDetector.py
└── yolov5.pt 

- Authors: Group A34_A_Gr_C
    - Diogo Nunes <up201808546@up.pt> 
    - João Rocha <up201806261@up.pt>
- Course: M.EIC 2022/2023
- University: FEUP