from pose_estimation.process.image import ImagePoseDetection
from pose_estimation.process.video import VideoPoseDetection
from pose_estimation.process.livestream import LiveStreamPoseDetection
from tqdm import tqdm

def main() -> int:

    #ipd = ImagePoseDetection()
    #results = ipd.classify("pose_estimation/images/inputs/topview.jpeg")
    #ipd.interpret_results(results)

    videos = ["forehand_unknown", "forehand_zhang_jike", "backhand_fan_zhendong", "backhand_harimoto", "backhand_lily_zhang", "backhand_samson_dubina", "forehand_liang_jingkun", "forehand_malong"]
    
    for video in tqdm(videos):
        print(video)
        vpd = VideoPoseDetection()
        results = vpd.classify(input_path=f"pose_estimation/videos/inputs/{video}.mp4",
                            output_path=f"pose_estimation/videos/outputs/{video}.mp4",
                            artifact_path=video)

    #lpd = LiveStreamPoseDetection()
    #results = lpd.classify()
    
    return 0

if __name__ == "__main__":
    main()