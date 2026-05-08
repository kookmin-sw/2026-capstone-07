python3 -m venv venv  
source venv/bin/activate  
pip install ultralytics opencv-python   


  python webcam_demo.py --device mps 
    python webcam_demo.py --device cpu --imgsz 480    