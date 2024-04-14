from pose_estimation.process.image import ImagePoseDetection
from pose_estimation.process.video import VideoPoseDetection
from pose_estimation.process.livestream import LiveStreamPoseDetection

def main() -> int:

    #ipd = ImagePoseDetection()
    #results = ipd.classify("pose_estimation/images/inputs/topview.jpeg")
    #ipd.interpret_results(results)
    
    vpd = VideoPoseDetection()
    results = vpd.classify(input_path="pose_estimation/videos/inputs/forehand_zhang_jike.mp4",
                           output_path="pose_estimation/videos/outputs/forehand_zhang_jike.mp4")

    #lpd = LiveStreamPoseDetection()
    #results = lpd.classify()
    
    return 0

if __name__ == "__main__":
    main()