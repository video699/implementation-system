from itertools import chain

import PIL
import cv2
import numpy as np
from annoy import AnnoyIndex
from fastai.vision.all import *
from fastai.vision.image import Transform
from fastcore.basics import fastuple

from video699.configuration import CONFIGURATION, LOGGER
from video699.interface import PageDetectorABC


def HSV(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, -1]


def crop(image):
    return image[:, 15:-12]


def sota(image, sigmaX=10):
    image = HSV(image)
    filtered = cv2.GaussianBlur(image, ksize=(6 * sigmaX + 1, 6 * sigmaX + 1), sigmaX=sigmaX)
    res = image.astype('int16') - filtered.astype('int16')
    res = np.interp(res, (res.min(), res.max()), (0, 255))
    return crop(res)


def open_image(fname, size=224):
    img = PIL.Image.open(fname)
    img = img.resize((size, size))
    t = torch.Tensor(np.array(img))
    return t.float() / 255.0


class SiameseImage(fastuple):
    def show(self, ctx=None, **kwargs):
        if len(self) > 2:
            img1, img2, similarity = self
        t1, t2 = img1, img2
        line = t1.new_zeros(t1.shape[0], t1.shape[1])
        return show_image(torch.cat([t1, line, t2], dim=1), title=similarity, ctx=ctx, cmap='gray', **kwargs)


def get_slide_path(screen_path):
    return Path('slides_sota') / screen_path.name


class SiameseTransform(Transform):
    def __init__(self, files, splits):
        self.train = [files[index] for index in splits[0]]
        self.valid = [files[index] for index in splits[1]]
        uniques = list(set(["-".join(file.name.split('-')[:2]) for file in files]))
        self.lecture2files = {unique: [f for f in files if unique in str(f)] for unique in uniques}

    def encodes(self, f):
        f2, same = self._draw(f)
        img1, img2 = open_image(f), open_image(f2)
        return SiameseImage(img1, img2, same)

    def _draw(self, f):
        same = random.random() < 0.5
        if same:
            return get_slide_path(f), same
        else:
            candidate = False
            while not candidate:
                file = get_slide_path(random.choice(self.lecture2files["-".join(f.name.split('-')[:2])]))
                if file.name != f.name:
                    candidate = True
            return get_slide_path(random.choice(self.lecture2files["-".join(f.name.split('-')[:2])])), same


class SiameseModel(Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder, self.head = encoder, head

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        ftrs = torch.cat([self.encoder(x1), self.encoder(x2)], dim=1)
        return self.head(ftrs)


class PyTorchSiamesePageDetector(PageDetectorABC):
    def __init__(self, documents):
        annoy_n_trees = CONFIGURATION.getint('annoy_n_trees')
        num_dense_units = CONFIGURATION.getint('num_dense_units')
        model = load_learner(Path('/home/xbankov/implementation-system/video699/page/restoration/models/export.pkl'))

        LOGGER.debug('Building an ANNOY index with {} trees'.format(annoy_n_trees))
        annoy_index = AnnoyIndex(num_dense_units, metric='euclidean')
        pages = dict()
        for page_index, (page, page_features) in enumerate(
                model.get_page_features(chain(*documents))
        ):
            annoy_index.add_item(page_index, page_features)
            pages[page_index] = page
        annoy_index.build(annoy_n_trees)

        self._annoy_index = annoy_index
        self._model = model
        self._pages = pages

    def detect(self, frame, appeared_screens, existing_screens, disappeared_screens):
        annoy_search_k = CONFIGURATION.getint('annoy_search_k')
        num_nearest_pages = CONFIGURATION.getint('num_nearest_pages')

        annoy_index = self._annoy_index
        model = self._model
        pages = self._pages

        detected_pages = {}
        screens = set(screen for screen, _ in chain(appeared_screens, existing_screens))
        for screen, screen_features in model.get_screen_features(screens):
            page_indices, page_distances = annoy_index.get_nns_by_vector(
                screen_features,
                num_nearest_pages,
                search_k=annoy_search_k,
                include_distances=True,
            )
            matches, = model.threshold_distances(page_distances).nonzero()
            if len(matches):
                closest_matching_page = pages[page_indices[matches[0]]]
            else:
                closest_matching_page = None
            detected_pages[screen] = closest_matching_page

        return detected_pages


def siamese_splitter(model):
    return [params(model.encoder), params(model.head)]


def loss_func(out, targ):
    return BCELossFlat()(out, targ.long())
