import cv2
import os

# Main folder where .mp4 videos are located
video_dir = "videos"
# Output folder for extracted frames
output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

# Walk through subdirectories like videos/fake, videos/real
for root, dirs, files in os.walk(video_dir):
    for filename in files:
        if filename.endswith(".mp4"):
            video_path = os.path.join(root, filename)
            vid = cv2.VideoCapture(video_path)
            ret, frame = vid.read()
            if ret:
                # Create output filename like: vid84__frame0.jpg
                base_name = os.path.splitext(filename)[0]
                frame_filename = f"{base_name}__frame0.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                print(f"✅ Saved first frame from {filename} → {frame_filename}")
            else:
                print(f"❌ Could not read from {filename}")
            vid.release()

