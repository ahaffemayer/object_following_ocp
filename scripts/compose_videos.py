import argparse
from contextlib import contextmanager
from pathlib import Path

import cv2
import ffmpeg
import numpy as np
import yaml
from tqdm import tqdm


@contextmanager
def VideoCapture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


class VideoReader:
    """Read frames from a video file using opencv, resizing to target size."""

    def __init__(self, path, target_width=None, target_height=None):
        self.path = path
        self.target_width = target_width
        self.target_height = target_height
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file {path}")

    def __enter__(self):
        return self

    def release(self):
        self.cap.release()

    def __exit__(self, *args):
        self.release()

    def get(self, prop):
        return self.cap.get(prop)

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.target_width and self.target_height:
            h, w = frame.shape[:2]
            if w != self.target_width or h != self.target_height:
                frame = cv2.resize(
                    frame,
                    (self.target_width, self.target_height),
                    interpolation=cv2.INTER_CUBIC,
                )
        return frame

    def set_frame(self, frame_number):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)


class VideoWriter:
    def __init__(
        self, path, fps, width, height, pix_fmt="yuv420p", vcodec="libx264", crf=28
    ):
        self.path = path
        self.width = width
        self.height = height

        pad_width = "if(gt(mod(iw,2),0),iw+1,iw)"
        pad_height = "if(gt(mod(ih,2),0),ih+1,ih)"

        self.process = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s="{}x{}".format(width, height),
                framerate=fps,
            )
            .filter("pad", w=pad_width, h=pad_height)
            .output(str(path), pix_fmt=pix_fmt, vcodec=vcodec, crf=crf)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    def write(self, frame):
        self.process.stdin.write(frame.tobytes())

    def write_batch(self, frames):
        for frame in frames:
            self.write(frame)

    def close(self):
        self.process.stdin.close()
        self.process.wait()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def ease_in_ease_out(x, p=1.5):
    return (x**p) / (x**p + (1 - x) ** p)


def get_num_frames(path):
    with VideoCapture(str(path)) as cap:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def get_fps(path):
    with VideoCapture(str(path)) as cap:
        return cap.get(cv2.CAP_PROP_FPS)


# ---------------------------------------------------------------------------
# YAML parsing: nested block structure -> flat (path, (row, col)) list
# ---------------------------------------------------------------------------


def parse_video_node(node, row_offset, col_offset, grid_size):
    """
    Recursively parse a YAML node into a list of (path, (row, col)) pairs.

    A node is either:
      {'path': '...'}              -> single video at (row_offset, col_offset), grid_size must be 1
      {'block': [n0, n1, n2, n3]} -> 2x2 arrangement of 4 sub-nodes, each occupying
                                     a (grid_size/2) x (grid_size/2) sub-region:
                                       n0=top-left, n1=top-right, n2=bottom-left, n3=bottom-right

    grid_size: number of cells this node occupies (must be a power of 2).
    """
    if "path" in node:
        assert grid_size == 1, (
            f"Path node found where grid_size={grid_size} expected a block: {node['path']}"
        )
        return [(Path(node["path"]), (row_offset, col_offset))]

    elif "block" in node:
        children = node["block"]
        assert len(children) == 4, (
            f"Each 'block' must have exactly 4 children, got {len(children)}"
        )
        half = grid_size // 2
        assert half * 2 == grid_size, f"grid_size must be a power of 2, got {grid_size}"

        # Children in reading order: top-left, top-right, bottom-left, bottom-right
        offsets = [
            (row_offset, col_offset),  # top-left
            (row_offset, col_offset + half),  # top-right
            (row_offset + half, col_offset),  # bottom-left
            (row_offset + half, col_offset + half),  # bottom-right
        ]
        result = []
        for child, (r, c) in zip(children, offsets):
            result.extend(parse_video_node(child, r, c, half))
        return result

    else:
        raise ValueError(f"Each video node must have 'path' or 'block', got: {node}")


def load_videos_from_config(config):
    """
    Parse config and return:
      video_paths     : list of Path in grid-fill order
      video_positions : list of (row, col) matching video_paths
      max_grid        : total grid size (e.g. 8 for 8x8)
    """
    grid_seq = config["grid_seq"]
    max_grid = max(grid_seq)
    raw_videos = config["videos"]

    # Wrap top-level list in a synthetic root block
    root = {"block": raw_videos}
    pairs = parse_video_node(root, 0, 0, max_grid)

    video_paths = [p for p, _ in pairs]
    video_positions = [pos for _, pos in pairs]
    return video_paths, video_positions, max_grid


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def compose_grid(video_readers, video_positions, grid_size, cell_width, cell_height):
    """Compose one frame from all readers into a full grid canvas."""
    n_videos = grid_size**2
    output_frame = np.zeros(
        (cell_height * grid_size, cell_width * grid_size, 3), dtype=np.uint8
    )
    for video_idx in range(n_videos):
        vr = video_readers[video_idx]
        frame = vr.read()
        if frame is None:
            vr.set_frame(0)
            frame = vr.read()
            assert frame is not None
        row, col = video_positions[video_idx]
        output_frame[
            row * cell_height : (row + 1) * cell_height,
            col * cell_width : (col + 1) * cell_width,
            :,
        ] = frame
    return output_frame


def write_single_video(video_reader, video_writer, num_frames):
    """Write num_frames of a single video fullscreen."""
    for _ in tqdm(range(num_frames), desc="Single video", ncols=80):
        frame = video_reader.read()
        if frame is None:
            video_reader.set_frame(0)
            frame = video_reader.read()
        video_writer.write(frame)


def write_dezoom(
    video_readers,
    video_writer,
    video_positions,
    from_grid,
    to_grid,
    num_frames,
    zoom_num_frames=40,
    cell_width=None,
    cell_height=None,
):
    """
    Animate a dezoom from from_grid x from_grid to to_grid x to_grid.

    The top-left from_grid x from_grid region of the composed to_grid canvas is
    cropped and upscaled to fill the output frame. The crop window then gradually
    expands to reveal the full grid. After the transition, hold on the full grid.
    """
    width = video_writer.width
    height = video_writer.height
    cw = cell_width or width
    ch = cell_height or height

    full_w = cw * to_grid
    full_h = ch * to_grid
    zoom_w = cw * from_grid
    zoom_h = ch * from_grid

    print(
        f"\nDezoom {from_grid}x{from_grid} -> {to_grid}x{to_grid}  "
        f"({zoom_num_frames} transition + {num_frames} hold frames)"
    )

    # Transition frames
    for frame_idx in tqdm(
        range(zoom_num_frames), desc=f"Dezoom {from_grid}->{to_grid}", ncols=80
    ):
        alpha = ease_in_ease_out(frame_idx / zoom_num_frames, p=1.5)
        grid_frame = compose_grid(video_readers, video_positions, to_grid, cw, ch)
        w = int((1 - alpha) * zoom_w + alpha * full_w)
        h = int((1 - alpha) * zoom_h + alpha * full_h)
        cropped = grid_frame[0:h, 0:w]
        resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_CUBIC)
        video_writer.write(resized)

    # Hold frames
    for _ in tqdm(range(num_frames), desc=f"Hold {to_grid}x{to_grid}", ncols=80):
        grid_frame = compose_grid(video_readers, video_positions, to_grid, cw, ch)
        resized = cv2.resize(grid_frame, (width, height), interpolation=cv2.INTER_CUBIC)
        video_writer.write(resized)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    grid_seq = config["grid_seq"]
    assert grid_seq[0] == 1, "grid_seq must start with 1"

    video_paths, video_positions, max_grid = load_videos_from_config(config)
    needed = max_grid**2
    assert len(video_paths) == needed, (
        f"Expected {needed} videos for {max_grid}x{max_grid} grid, got {len(video_paths)}"
    )

    width = args.width
    height = args.height
    avg_fps = get_fps(video_paths[0])
    duration_frames = get_num_frames(video_paths[0])

    print(f"Grid sequence : {grid_seq}")
    print(f"Max grid      : {max_grid}x{max_grid} ({needed} videos)")
    print(f"Output        : {args.output_file}  {width}x{height} @ {avg_fps:.2f} fps")
    print(f"First video   : {duration_frames} frames")

    # Print position map for verification
    print(f"\nVideo grid layout ({max_grid}x{max_grid}):")
    grid_display = [["?" for _ in range(max_grid)] for _ in range(max_grid)]
    for idx, (path, (r, c)) in enumerate(zip(video_paths, video_positions)):
        grid_display[r][c] = f"{idx:2d}:{path.name[:10]}"
    col_w = max(len(cell) for row in grid_display for cell in row) + 1
    for row in grid_display:
        print("  " + " | ".join(cell.ljust(col_w) for cell in row))

    video_readers = [
        VideoReader(p, target_width=width, target_height=height) for p in video_paths
    ]

    with VideoWriter(args.output_file, avg_fps, width, height, crf=args.crf) as writer:
        # Step 1: first video fullscreen
        print(f"\nWriting first video fullscreen ({duration_frames} frames)...")
        write_single_video(video_readers[0], writer, duration_frames)

        # Steps 2..N: dezoom transitions
        for from_g, to_g in zip(grid_seq[:-1], grid_seq[1:]):
            for vr in video_readers:
                vr.set_frame(0)
            write_dezoom(
                video_readers,
                writer,
                video_positions,
                from_grid=from_g,
                to_grid=to_g,
                num_frames=args.hold_num_frames,
                zoom_num_frames=args.zoom_num_frames,
                cell_width=width,
                cell_height=height,
            )

    for reader in video_readers:
        reader.release()

    print(f"\nDone! Written to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    parser.add_argument(
        "--output_file", type=str, default="merged.mp4", help="Output video file"
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--zoom_num_frames",
        type=int,
        default=40,
        help="Number of frames for each dezoom transition",
    )
    parser.add_argument(
        "--hold_num_frames",
        type=int,
        default=150,
        help="Number of frames to hold on each grid after transition",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=15,
        help="CRF for ffmpeg encoding (lower = better quality)",
    )
    args = parser.parse_args()
    main(args)
