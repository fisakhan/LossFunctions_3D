import os
import cv2
import numpy as np


def combine_videos_horizontally(
    video_paths, output_path="combined_loss_comparison.mp4", fps=5
):
    # 1. Open all input video streams
    caps = [cv2.VideoCapture(p) for p in video_paths]

    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_paths[i]}")
            return

    # 2. Extract dimensions from the first video frame
    frame_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video dimensions: width = (single_width * 3), height = single_height + title_header
    header_height = 80
    combined_width = frame_width * len(video_paths)
    combined_height = frame_height + header_height

    # 3. Configure VideoWriter for output MP4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_path, fourcc, fps, (combined_width, combined_height)
    )

    titles = ["Standard Softmax", "CurricularFace Loss", "UniFace (UCE) Loss"]
    panel_colors = [
        (50, 50, 200),
        (50, 180, 50),
        (200, 100, 50),
    ]  # Subtle accent colors

    print("Combining video files side-by-side...")
    frame_count = 0

    while True:
        frames = []
        read_success = True

        # Read one frame from each video
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                read_success = False
                break
            frames.append(frame)

        if not read_success:
            break

        # 4. Create top header banner
        header = (
            np.ones((header_height, combined_width, 3), dtype=np.uint8) * 245
        )  # Off-white background

        # Render panel titles onto top header
        for i, title in enumerate(titles):
            x_offset = i * frame_width + (frame_width // 2)

            # Draw colored title text centered above each video panel
            text_size = cv2.getTextSize(
                title, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
            )[0]
            text_x = x_offset - (text_size[0] // 2)
            text_y = (header_height // 2) + (text_size[1] // 2)

            cv2.putText(
                header,
                title,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                panel_colors[i],
                2,
                cv2.LINE_AA,
            )

            # Draw vertical subtle divider between panels
            if i > 0:
                cv2.line(
                    header,
                    (i * frame_width, 10),
                    (i * frame_width, header_height - 10),
                    (200, 200, 200),
                    2,
                )

        # 5. Concatenate video frames horizontally
        video_row = np.hstack(frames)

        # Draw vertical lines separating the video panels below the header
        for i in range(1, len(video_paths)):
            cv2.line(
                video_row,
                (i * frame_width, 0),
                (i * frame_width, frame_height),
                (180, 180, 180),
                2,
            )

        # 6. Combine header and video row vertically
        combined_frame = np.vstack((header, video_row))

        out.write(combined_frame)
        frame_count += 1

    # Clean up resources
    for cap in caps:
        cap.release()
    out.release()

    print(f"\nSuccess! Processed {frame_count} frames.")
    print(f"Combined video saved to: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    input_videos = [
        "softmax_evolution.mp4",
        "curricularface_evolution.mp4",
        "uniface_evolution.mp4",
    ]

    combine_videos_horizontally(
        video_paths=input_videos,
        output_path="combined_loss_comparison.mp4",
        fps=5,
    )
