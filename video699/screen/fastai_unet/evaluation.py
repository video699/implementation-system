from tqdm import tqdm
from video699.screen.fastai_unet.postprocessing import iou
import numpy as np
from matplotlib import pyplot as plt


def all_video_statistics(videos, actual_detector):
    sizes = []
    ratios = []
    for video in videos:
        size, ratio = single_video_statistics(video, actual_detector)
        sizes.extend(size)
        ratios.extend(ratio)
    return sizes, ratios


def single_video_statistics(video, actual_detector):
    sizes = []
    ratios = []
    for frame in tqdm(video):
        actual_screens = actual_detector.detect(frame)
        for screen in actual_screens:
            sizes.append(screen.coordinates.area)
            ratios.append(screen.coordinates.height / screen.coordinates.width)
    return sizes, ratios


def evaluate(actuals, preds):
    assert len(actuals) == len(preds)
    wrong_screen_count_frames = []
    ious = []
    really_bad_ious = []
    for index, zipped in enumerate(zip(actuals, preds)):
        frame_ious = []
        actual_screens, pred_screens = zipped
        actual_screens = sorted(actual_screens, key=lambda screen: screen.coordinates.top_left[0])
        pred_screens = sorted(pred_screens, key=lambda screen: screen.coordinates.top_left[0])
        if len(actual_screens) != len(pred_screens):
            wrong_screen_count_frames.append(index)
        else:
            for screenA, screenB in zip(actual_screens, pred_screens):
                score = iou(screenA, screenB)
                frame_ious.append(score)

        if len(frame_ious) > 0:
            score = np.array(frame_ious).mean()
        else:
            score = np.nan
        ious.append(score)
        if score < 0.92:
            really_bad_ious.append(index)
    return wrong_screen_count_frames, ious, really_bad_ious


def all_videos_eval(videos, actual_detector, pred_detector):
    all_wrong_screen_count_frames, all_ious, all_really_bad_ious = [], [], []
    for video in videos:
        wrong_screen_count_frames, ious, really_bad_ious = single_video_eval(video, actual_detector,
                                                                             pred_detector)
        all_wrong_screen_count_frames.extend(wrong_screen_count_frames)
        all_ious.extend(ious)
        all_really_bad_ious.extend(really_bad_ious)
    return all_wrong_screen_count_frames, all_ious, all_really_bad_ious


def single_video_eval(video, actual_detector, pred_detector):
    wrong_screen_count_frames = []
    ious = []
    really_bad_ious = []
    for frame in tqdm(video):
        frame_ious = []
        actual_screens = sorted(actual_detector.detect(frame),
                                key=lambda screen: screen.coordinates.top_left[0])
        pred_screens = sorted(pred_detector.detect(frame),
                              key=lambda screen: screen.coordinates.top_left[0])
        if len(actual_screens) != len(pred_screens):
            wrong_screen_count_frames.append(frame)
            # Think about what to do !!!
        else:
            for screenA, screenB in zip(actual_screens, pred_screens):
                score = iou(screenA, screenB)
                frame_ious.append(score)

        if len(frame_ious) > 0:
            score = np.array(frame_ious).mean()
        else:
            score = np.nan
        ious.append(score)
        if score < 0.92:
            really_bad_ious.append(frame)
    return wrong_screen_count_frames, ious, really_bad_ious


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))


def single_frame_visualization(frame, actual_detector, pred_detector):
    actual_screens = sorted(actual_detector.detect(frame),
                            key=lambda screen: screen.coordinates.top_left[0])
    pred_screens = sorted(pred_detector.detect(frame),
                          key=lambda screen: screen.coordinates.top_left[0])

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 4), sharex='row', sharey='row')
    fig.tight_layout()
    for screen in actual_screens:
        polygon = screen.coordinates._polygon
        axes[0].set_title("Actual")
        axes[0].plot(*polygon.exterior.xy, c='tab:orange')
        axes[2].plot(*polygon.exterior.xy, c='tab:orange', label="Actual")

    for screen in pred_screens:
        polygon = screen.coordinates._polygon
        axes[1].set_title("Prediction")
        axes[1].plot(*polygon.exterior.xy, c='tab:blue')
        axes[2].plot(*polygon.exterior.xy, c='tab:blue', label="Predicted")

    axes[2].set_title("Combined")
    legend_without_duplicate_labels(axes[2])

    axes[3].set_title("Original")
    axes[3].imshow(frame.image)
    plt.show()


def all_frames_visualization(frames, actual_detector, pred_detector):
    for frame in frames:
        single_frame_visualization(frame, actual_detector, pred_detector)
