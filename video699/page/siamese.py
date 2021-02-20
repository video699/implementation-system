# -*- coding: utf-8 -*-

r"""This module implements a page detector that matches feature vectors extracted from document page
image data with feature vectors extracted from projection screen image data using a Siamese deep
convolutional neural network. Related classes and functions are also implemented.

"""


import pickle
from hashlib import md5
from itertools import chain
from logging import getLogger
from math import ceil
import os
from random import shuffle

from annoy import AnnoyIndex
import cv2 as cv
import keras.backend as K
from keras.layers import concatenate, Conv2D, Dense, Flatten, Input, Lambda, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.utils import Sequence
import numpy as np
from npstreams.iter_utils import last
from npstreams.stats import _ivar

from video699.common import get_batches
from video699.configuration import get_configuration
from video699.interface import PageDetectorABC
from video699.video.annotated import get_videos, AnnotatedSampledVideoScreenDetector


LOGGER = getLogger(__name__)
ALL_VIDEOS = set(get_videos().values())
CONFIGURATION = get_configuration()['KerasSiamesePageDetector']
RESOURCES_PATHNAME = os.path.join(os.path.dirname(__file__), 'siamese')
TRAINING_SCREEN_DETECTOR = AnnotatedSampledVideoScreenDetector(beyond_bounds=False)
VALIDATION_SCREEN_DETECTOR = AnnotatedSampledVideoScreenDetector()


def feature_tensor_l2_distances(screen_features, page_features):
    r"""The :math:`L_2` distances between a pair of feature tensors.

    Parameters
    ----------
    screen_features : tensor
        A tensor :math:`\mathbf X` of features :math:`\mathbf x` extracted from a batch of screen
        images.
    page_features : tensor
        A tensor :math:`\mathbf Y` of features :math:`\mathbf y` extracted from a batch of page
        images.

    Returns
    -------
    l2_distances : tensor
        A tensor of :math:`L_2` distances :math:`\lVert\mathbf{x - y}\rVert_2`.

    """

    distances = K.sqrt(K.sum((screen_features - page_features)**2, axis=-1))
    return K.expand_dims(distances)


class _ImageMoments(object):
    """Statistical moments extracted from lit projection screens and from document pages.

    Parameters
    ----------
    videos : iterable of VideoABC
        The video in which we will detect lit projection screens.
    screen_detector : ScreenDetectorABC
        The screen detector that will be used to detect lit projection screens in the video frames.
        We will extract statistical moments from the screens.
    documents : iterable of DocumentABC
        The documents from whose pages we will extract statistical moments.

    Attributes
    ----------
    mean_screen : np.array
        The mean preprocessed grayscale screen image.
    inverse_screen_std : np.array
        The inverse standard deviation of a preprocessed grayscale screen image.
    mean_page : np.array
        The mean preprocessed grayscale page image.
    inverse_page_std : np.array
        The inverse standard deviation of a preprocessed grayscale page image.
    """

    def __init__(self, videos, screen_detector, documents):
        LOGGER.info('Extracting projection screen moments')
        mean_screen, squared_mean_screen, _ = last(_ivar(
            map(
                _preprocess_image,
                (
                    screen
                    for video in videos
                    for frame in video
                    for screen in screen_detector.detect(frame)
                ),
            ),
        ))
        inverse_screen_std = 1 / np.sqrt(squared_mean_screen - mean_screen**2)
        inverse_screen_std[np.isinf(inverse_screen_std)] = 0.0

        LOGGER.info('Extracting document page moments')
        mean_page, squared_mean_page, _ = last(_ivar(
            map(
                _preprocess_image,
                (
                    page
                    for document in documents
                    for page in document
                ),
            ),
        ))
        inverse_page_std = 1 / np.sqrt(squared_mean_page - mean_page**2)
        inverse_page_std[np.isinf(inverse_page_std)] = 0.0

        LOGGER.info('Done extracting statistical moments')

        self.mean_screen = mean_screen
        self.inverse_screen_std = inverse_screen_std
        self.mean_page = mean_page
        self.inverse_page_std = inverse_page_std


class _AnnotatedImagePairs(Sequence):
    """Human-annotated pairs of screens and pages produced from a set of human-annotated videos.

    For every combination of a screen in the set of human-annotated videos and a document page, a
    pair is produced. Every matching pair of a screen and a page is assigned a classification label
    of 0, whereas non-matching pairs are assigned the label of 1. For every screen, the non-matching
    pairs and the matching pairs have an equal sum of their weights to offset the class imbalance
    that favors non-matching pairs.

    Note
    ----
    The choice of labels is significant. The sigmoid function predicts the label from the
    :math:`L_2` distance between feature vectors.  Close feature vectors will have a distance close
    to zero, which the sigmoid function will transform to the label 0. Distant feature vectors will
    have a large distance, which the sigmoid function will transform to the label 1.

    Parameters
    ----------
    videos : set of AnnotatedSampledVideo
        A set of human-annotated videos that will be used to produce pars of projection screens, and
        document pages.
    moments : _ImageMoments
        Statistical moments used to normalize image data.
    screen_detector : ScreenDetectorABC
        A screen detector that will be used to detect screens in the videos.

    """

    def __init__(self, videos, moments, screen_detector):
        matching_pair_base_weight = CONFIGURATION.getfloat('matching_pair_base_weight')
        distant_nonmatching_pair_weight = CONFIGURATION.getfloat('distant_nonmatching_pair_weight')
        close_nonmatching_pair_weight = CONFIGURATION.getfloat('close_nonmatching_pair_weight')
        CONFIGURATION.getfloat('matching_pair_base_weight')
        samples = []

        for video in videos:
            documents = set(video.documents.values())
            for frame in video:
                for screen in screen_detector.detect(frame):
                    fully_matching_pages, incrementally_matching_pages, _ = screen.matching_pages()
                    matching_pages = set(fully_matching_pages) | set(incrementally_matching_pages)
                    matching_documents = set(page.document for page in matching_pages)
                    matching_weights = matching_pair_base_weight

                    for document in matching_documents:
                        for page in document:
                            if page not in matching_pages:
                                label = 1
                                weight = close_nonmatching_pair_weight
                                samples.append((screen, page, label, weight))
                                matching_weights += weight

                    for document in documents - matching_documents:
                        for page in document:
                            label = 1
                            weight = distant_nonmatching_pair_weight
                            samples.append((screen, page, label, weight))
                            matching_weights += weight

                    for page in fully_matching_pages:
                        label = 0
                        weight = matching_weights / len(matching_pages)
                        samples.append((screen, page, label, weight))

        inverse_weight_norm = len(samples) / sum(weight for screen, page, label, weight in samples)
        shuffle(samples)

        LOGGER.info('Produced a dataset containing {} annotated image pairs'.format(len(samples)))

        self._moments = moments
        self._inverse_weight_norm = inverse_weight_norm
        self._samples = samples

    def shuffle(self):
        """Shuffles the pairs of screens and pages.

        """

        samples = self._samples
        shuffle(samples)

    def __len__(self):
        training_batch_size = CONFIGURATION.getint('training_batch_size')
        samples = self._samples
        return ceil(len(samples) / training_batch_size)

    def __getitem__(self, idx):
        training_batch_size = CONFIGURATION.getint('training_batch_size')
        image_dtype = np.__dict__[CONFIGURATION['image_dtype']]
        image_width = CONFIGURATION.getint('image_width')
        image_height = CONFIGURATION.getint('image_height')

        moments = self._moments
        inverse_weight_norm = self._inverse_weight_norm
        samples = self._samples

        effective_batch_size = min(training_batch_size, len(samples) - idx * training_batch_size)
        screen_images = np.empty(
            (effective_batch_size, image_height, image_width, 1),
            dtype=image_dtype,
        )
        page_images = np.empty(
            (effective_batch_size, image_height, image_width, 1),
            dtype=image_dtype,
        )
        targets = []
        sample_weights = []
        for batch_index in range(effective_batch_size):
            sample_index = idx * training_batch_size + batch_index
            screen, page, label, weight = samples[sample_index]

            standardized_screen_image = (
                _preprocess_image(screen) - moments.mean_screen
            ) * moments.inverse_screen_std
            screen_images[batch_index, :, :, 0] = standardized_screen_image

            standardized_page_image = (
                _preprocess_image(page) - moments.mean_page
            ) * moments.inverse_page_std
            page_images[batch_index, :, :, 0] = standardized_page_image

            targets.append(label)
            sample_weights.append(weight)

        inputs = [screen_images, page_images]
        return inputs, targets, np.asfarray(sample_weights) * inverse_weight_norm


class _KerasSiameseNeuralNetwork(object):
    """A Siamese convolutional neural network trained on a set of human-annotated videos.

    Notes
    -----
    All human-annotated videos that are not part of the training set will be used as the validation
    set. The validation loss and accuracy will be recorded, but they will not be used to influence
    the training. Therefore, the validation set can still be used as the test set in subsequent
    evaluation.

    Parameters
    ----------
    training_videos : set of AnnotatedSampledVideo or None, optional
        The human-annotated videos that will be used to train the Siamese deep convolutional neural
        network.  If ``None`` or unspecified, all human-annotated videos will
        be used as the training set.
    make_persistent : bool, optional
        Whether the neural network will be persistently stored for future reuse.  When unspecified,
        or ``False``, a neural network will not be stored, but an existing neural network will
        nevertheless be loaded.

    Attributes
    ----------
    regression_model : Keras.models.Model
        A deep convolutional neural network trained on the training set of human-annotated videos.
        Given a screen, or a page image, the neural network extracts deep image features.
    thresholding_model : Keras.models.Model
        A Siamese deep convolutional neural network trained on the training set of human-annotated
        videos. Given an :math:`L_2` distance between screen image and page image features, the
        neural network predicts a class label.
    training_moments : _ImageMoments
        Statistical moments extracted from the training videos.
    training_history : dict
        The `history` attribute of the :class:`keras.callbacks.History` object produced during the
        training.

    """

    def __init__(self, training_videos=None, make_persistent=True):
        if training_videos is None:
            training_videos = ALL_VIDEOS

        image_width = CONFIGURATION.getint('image_width')
        image_height = CONFIGURATION.getint('image_height')
        min_training_accuracy = CONFIGURATION.getfloat('min_training_accuracy')
        num_dense_units = CONFIGURATION.getint('num_dense_units')
        filter_width = CONFIGURATION.getint('filter_width')
        filter_height = CONFIGURATION.getint('filter_height')
        filter_size = (filter_height, filter_width)
        num_top_filters = CONFIGURATION.getint('num_top_filters')
        num_bottom_filters = CONFIGURATION.getint('num_bottom_filters')
        maxpool_width = CONFIGURATION.getint('maxpool_width')
        maxpool_height = CONFIGURATION.getint('maxpool_height')
        maxpool_size = (maxpool_height, maxpool_width)
        learning_rate = CONFIGURATION.getfloat('learning_rate')

        if training_videos is None:
            training_videos = ALL_VIDEOS

        if training_videos == ALL_VIDEOS:
            model_dirname = 'pretrained'
        else:
            model_dirname_hash = md5()
            for video_filename in sorted(video.filename for video in training_videos):
                model_dirname_hash.update(video_filename.encode())
            model_dirname = model_dirname_hash.hexdigest()

        model_pathname = os.path.join(RESOURCES_PATHNAME, model_dirname)
        classification_model_pathname = os.path.join(model_pathname, 'classification_model.h5')
        training_moments_pathname = os.path.join(model_pathname, 'training_moments.pkl')
        training_history_pathname = os.path.join(model_pathname, 'training_history.pkl')
        format_version_pathname = os.path.join(model_pathname, 'format_version')

        screen_input = Input(shape=(image_height, image_width, 1))
        page_input = Input(shape=(image_height, image_width, 1))
        thresholding_input = Input(shape=(1,))

        convnet_model = Sequential([
            Conv2D(num_top_filters, filter_size, activation='relu', padding='same'),
            MaxPooling2D(pool_size=maxpool_size),
            Conv2D(num_top_filters, filter_size, activation='relu', padding='same'),
            MaxPooling2D(pool_size=maxpool_size),
            Conv2D(num_top_filters, filter_size, activation='relu', padding='same'),
            MaxPooling2D(pool_size=maxpool_size),
            Conv2D(num_top_filters, filter_size, activation='relu', padding='same'),
            MaxPooling2D(pool_size=maxpool_size),
            Conv2D(num_bottom_filters, filter_size, activation='relu', padding='same'),
            Conv2D(num_bottom_filters, filter_size, activation='relu', padding='same'),
            MaxPooling2D(pool_size=maxpool_size),
            Conv2D(num_bottom_filters, filter_size, activation='relu', padding='same'),
            Conv2D(num_bottom_filters, filter_size, activation='relu', padding='same'),
            MaxPooling2D(pool_size=maxpool_size),
            Conv2D(num_bottom_filters, filter_size, activation='relu', padding='same'),
            Conv2D(num_bottom_filters, filter_size, activation='relu', padding='same'),
            Flatten(),
            Dense(num_dense_units, activation='relu'),
        ])

        distance_layer = Lambda(
            lambda inputs: feature_tensor_l2_distances(
                inputs[:, num_dense_units:],
                inputs[:, :num_dense_units],
            ),
        )
        thresholding_layer = Dense(1, activation='sigmoid')
        dense_model = Sequential([distance_layer, thresholding_layer])

        screen_convnet_tensor = convnet_model(screen_input)
        regression_model = Model(inputs=screen_input, outputs=screen_convnet_tensor)

        page_convnet_tensor = convnet_model(page_input)
        classification_tensor = dense_model(
            concatenate([screen_convnet_tensor, page_convnet_tensor]),
        )
        classification_model = Model(
            inputs=[screen_input, page_input],
            outputs=classification_tensor,
        )

        thresholding_tensor = thresholding_layer(thresholding_input)
        thresholding_model = Model(inputs=thresholding_input, outputs=thresholding_tensor)

        try:
            classification_model.load_weights(classification_model_pathname)

            with open(training_moments_pathname, 'rb') as f:
                training_moments = pickle.load(f)
            with open(training_history_pathname, 'rb') as f:
                training_history = pickle.load(f)
            with open(format_version_pathname, 'rt') as f:
                format_version = f.read()
                assert format_version == '1'

            training_accuracy = training_history['weighted_acc'][-1]
            if training_accuracy < min_training_accuracy:
                LOGGER.warning(
                    'The loaded model has an insufficient training accuracy of {} (< {})'.format(
                        training_accuracy,
                        min_training_accuracy,
                    ),
                )

            LOGGER.info('Loaded a model from {}'.format(model_pathname))
        except IOError:
            classification_model.compile(
                optimizer=SGD(lr=learning_rate),
                loss='binary_crossentropy',
                weighted_metrics=['accuracy'],
            )

            num_epochs = CONFIGURATION.getint('num_training_epochs')

            training_moments = _ImageMoments(
                training_videos,
                TRAINING_SCREEN_DETECTOR,
                (document for video in training_videos for document in video.documents.values()),
            )
            training_generator = _AnnotatedImagePairs(
                training_videos,
                training_moments,
                TRAINING_SCREEN_DETECTOR,
            )
            validation_videos = ALL_VIDEOS - training_videos
            training_kwargs = {'verbose': 0, 'epochs': num_epochs}
            if validation_videos:
                validation_generator = _AnnotatedImagePairs(
                    validation_videos,
                    training_moments,
                    VALIDATION_SCREEN_DETECTOR,
                )
                training_kwargs['validation_data'] = validation_generator

            training_accuracy = float('-inf')
            while training_accuracy < min_training_accuracy:
                LOGGER.info('Training a model for {} epochs'.format(num_epochs))
                training_history = classification_model.fit_generator(
                    training_generator,
                    **training_kwargs,
                ).history

                training_accuracy = training_history['weighted_acc'][-1]
                if training_accuracy < min_training_accuracy:
                    LOGGER.info(
                        'Rebuilding model with insufficient training accuracy of {} (< {})'.format(
                            training_accuracy,
                            min_training_accuracy,
                        ),
                    )
                    training_generator.shuffle()
                    session = K.get_session()
                    for layer in chain(dense_model.layers, convnet_model.layers):
                        if hasattr(layer, 'kernel_initializer'):
                            layer.kernel.initializer.run(session=session)

            if make_persistent:
                os.mkdir(model_pathname)
                classification_model.save_weights(classification_model_pathname)
                with open(training_moments_pathname, 'wb') as f:
                    pickle.dump(training_moments, f)
                with open(training_history_pathname, 'wb') as f:
                    pickle.dump(training_history, f)
                with open(format_version_pathname, 'wt') as f:
                    f.write('1')
                LOGGER.info('Stored a model in {}'.format(model_pathname))

        self.regression_model = regression_model
        self.thresholding_model = thresholding_model
        self.training_moments = training_moments
        self.training_history = training_history

    def get_screen_features(self, screens):
        """Extracts deep features from projection screen images.

        Parameters
        ----------
        screens : iterable of ScreenABC
            Projection screens from which we will extract deep image features.

        Yields
        ------
        screen : ScreenABC
            A projection screen.
        screen_features : np.array
            The deep features extracted from the projection screen image.

        """

        prediction_batch_size = CONFIGURATION.getint('prediction_batch_size')
        image_dtype = np.__dict__[CONFIGURATION['image_dtype']]
        image_width = CONFIGURATION.getint('image_width')
        image_height = CONFIGURATION.getint('image_height')

        regression_model = self.regression_model
        training_moments = self.training_moments

        for screen_batch in get_batches(screens, prediction_batch_size):
            standardized_screen_images = np.empty(
                (len(screen_batch), image_height, image_width, 1),
                dtype=image_dtype,
            )
            for screen_index, screen in enumerate(screen_batch):
                standardized_screen_image = (
                    _preprocess_image(screen) - training_moments.mean_screen
                ) * training_moments.inverse_screen_std
                standardized_screen_images[screen_index, :, :, 0] = standardized_screen_image
            for screen, features in zip(
                        screen_batch,
                        regression_model.predict(standardized_screen_images),
                    ):
                yield (screen, features)

    def get_page_features(self, pages):
        """Extracts deep features from document page images.

        Parameters
        ----------
        pages : iterable of PageABC
            Document pages from which we will extract deep image features.

        Yields
        ------
        page : PageABC
            A document page.
        page_features : np.array
            The deep features extracted from the document page image.

        """

        prediction_batch_size = CONFIGURATION.getint('prediction_batch_size')
        image_dtype = np.__dict__[CONFIGURATION['image_dtype']]
        image_width = CONFIGURATION.getint('image_width')
        image_height = CONFIGURATION.getint('image_height')

        regression_model = self.regression_model
        training_moments = self.training_moments

        for page_batch in get_batches(pages, prediction_batch_size):
            standardized_page_images = np.empty(
                (len(page_batch), image_height, image_width, 1),
                dtype=image_dtype,
            )
            for page_index, page in enumerate(page_batch):
                standardized_page_image = (
                    _preprocess_image(page) - training_moments.mean_page
                ) * training_moments.inverse_page_std
                standardized_page_images[page_index, :, :, 0] = standardized_page_image
            for page, features in zip(
                        page_batch,
                        regression_model.predict(standardized_page_images),
                    ):
                yield (page, features)

    def threshold_distances(self, distances):
        """Predicts whether projection screens and document pages match.

        Parameters
        ----------
        distances : array_like
            The :math:`L_2` distances between projection screen features and document page features.

        Returns
        -------
        thresholded_distances : array_like
            Whether the projection screens and document pages match.
        """

        significance_level = CONFIGURATION.getfloat('significance_level')

        thresholding_model = self.thresholding_model

        thresholded_distances = thresholding_model.predict(distances).ravel() < significance_level
        return thresholded_distances


def _preprocess_image(image):
    """Preprocesses an image to be used as an input to a Siamese deep convolutional neural network.

    Parameters
    ----------
    image : ImageABC
        An image to be preprocesses.

    Returns
    -------
    preprocessed_image : np.array
        The preprocessed grayscale image to be used as an input to a Siamese deep convolutional
        neural network.

    """

    image_dtype = np.__dict__[CONFIGURATION['image_dtype']]
    image_width = CONFIGURATION.getint('image_width')
    image_height = CONFIGURATION.getint('image_height')

    rgba_image = image.render(image_width, image_height)
    gray_image = cv.cvtColor(rgba_image, cv.COLOR_BGRA2GRAY)

    return gray_image.astype(image_dtype)


class KerasSiamesePageDetector(PageDetectorABC):
    r"""A page detector using approximate nearest neighbor search of deep image features.

    A convolutional neural network accepts images on the input and produces feature vectors on the
    output. During training, two “Siamese” copies of the convolutional neural network with shared
    weights are produced and topped with a lambda layer that computes the :math:`L_2` distance
    :math:`d` between the two output feature vectors, and with a dense one-unit layer with the
    sigmoid activation function :math:`S`. Pairs of screen and page images are fed to the Siamese
    network, and the binary cross-entropy loss function :math:`-(y\cdot\log(S(d)) + (1 -
    y)\cdot\log(1 - S(d)))` is used to evaluate how well the network classifies matching (:math:`y =
    0`), and non-matching (:math:`y = 1`) image pairs. This general Siamese architecture was first
    proposed by [BromleyEtAl94]_.

    Deep image features are extracted from the image data of the provided document pages using the
    trained convolutional neural network, and they are placed inside a vector database.  Deep image
    features are then extracted from the image data in a screen and nearest neighbors are retrieved
    from the vector database. The page images corresponding to the nearest neighbors are paired with
    the screen image and fed to the Siamese network. The document page with the nearest features
    that is predicted to match the screen is detected as the page shown in the screen. If none of
    the pages corresponding to the nearest neighbors is predicted to match the screen by the Siamese
    neural network, then no page is detected in the screen.

    .. [BromleyEtAl94] Bromley, Jane, et al. "Signature Verification using a 'Siamese' Time Delay
       Neural Network." *Advances in Neural Information Processing Systems*. 1994.

    Parameters
    ----------
    documents : set of DocumentABC
        The provided document pages.
    training_videos : set of AnnotatedSampledVideo or None, optional
        The human-annotated videos that will be used to train the Siamese deep convolutional neural
        network. When ``None`` or unspecified, all human-annotated videos will be used.
    """

    def __init__(self, documents, training_videos=None):
        if training_videos is None:
            training_videos = ALL_VIDEOS

        annoy_n_trees = CONFIGURATION.getint('annoy_n_trees')
        num_dense_units = CONFIGURATION.getint('num_dense_units')
        model = _KerasSiameseNeuralNetwork(training_videos)

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
