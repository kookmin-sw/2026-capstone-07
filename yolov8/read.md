python3 -m venv venv  
source venv/bin/activate  
pip install ultralytics opencv-python   

#Mac 실행
python webcam_demo.py --device mps 
#Window 실행
python webcam_demo.py --device cpu --imgsz 480    
