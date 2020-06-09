r"""This module implements an ensemble page detector that lets other page detectors vote.

"""


from ..interface import PageDetectorABC
from ..configuration import get_configuration


class EnsemblePageDetector(PageDetectorABC):
    r"""A page detector that uses a voting ensemble of page detectors.

    Parameters
    ----------
    ensemble : dict of (PageDetectorABC, float)
        The page detectors in the ensemble and their weights.
    quorum : float, optional
        Minimum vote necessary to detect a page. Default is 0.
    """

    def __init__(self, ensemble, quorum=0.0):
        self._ensemble = {
            page_detector: weight
            for page_detector, weight
            in ensemble.items()
            if weight > 0.0
        }
        self._quorum = quorum

    def detect(self, frame, appeared_screens, existing_screens, disappeared_screens):
        screens = (tuple(appeared_screens), tuple(existing_screens), tuple(disappeared_screens))
        ensemble = self._ensemble
        quorum = self._quorum

        votes = {}
        for page_detector, page_detector_weight in ensemble.items():
            for screen, detected_page in page_detector.detect(frame, *screens).items():
                if screen not in votes:
                    votes[screen] = {}
                if detected_page not in votes[screen]:
                    votes[screen][detected_page] = 0.0
                votes[screen][detected_page] += page_detector_weight

        detected_pages = {}
        for screen, pages in votes.items():
            detected_page, vote = max(pages.items(), key=lambda x: x[1])
            detected_pages[screen] = detected_page if vote >= quorum else None

        return detected_pages
