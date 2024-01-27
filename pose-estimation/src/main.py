from pose_estimation.process.image import ImagePoseDetection
from pose_estimation.process.video import VideoPoseDetection

def main() -> int:

    #ipd = ImagePoseDetection()
    #results = ipd.classify("pose_estimation/images/inputs/topview.jpeg")
    #ipd.interpret_results(results)
    
    vpd = VideoPoseDetection()
    results = vpd.classify("pose_estimation/videos/inputs/forehand_drive_malong.mp4")
    
    return 0

if __name__ == "__main__":
    main()