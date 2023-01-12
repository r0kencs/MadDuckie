README: InRoad Duckie Avoidance

- Operating System: Ubuntu 20.04
- Packages versions:
    - py-multihash: 0.2.0
    - Pillow: 8.3.2
    - pyglet: 1.5.11
    - testresources: should be installed, not in the requirements of gym-duckietown
    - numpy package: 1.23
    - PyTorch: 1.8.1+cu101

- Usage:
    - Install the required packages
        - git clone https://github.com/duckietown/gym-duckietown.git
        - cd gym-duckietown
        - pip3 install -e .
    - Install PyTorch
    - Check if all the packages above have the correct versions (latest versions might bring errors)
    - Access project's folder and give permissions to the main.py file (if needed):
        - chmod u+x main.py
    - Run the command:
        - ./main.py

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