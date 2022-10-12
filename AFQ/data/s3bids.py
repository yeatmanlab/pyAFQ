from io import BytesIO
import gzip
import tempfile

import s3fs
import boto3
from botocore import UNSIGNED
from botocore.client import Config

from dask import compute, delayed
from dask.diagnostics import ProgressBar

from pathlib import Path
import os
import os.path as op

import logging
import numpy as np
import pandas as pd
import json
from tqdm.auto import tqdm
import nibabel as nib

from bids import BIDSLayout
from AFQ.data.fetch import to_bids_description


# +----------------------------------------------------+
# | Begin S3BIDSStudy classes and supporting functions |
# +----------------------------------------------------+
def get_s3_client(anon=True):
    """Return a boto3 s3 client

    Global boto clients are not thread safe so we use this function
    to return independent session clients for different threads.

    Parameters
    ----------
    anon : bool
        Whether to use anonymous connection (public buckets only).
        If False, uses the key/secret given, or boto’s credential
        resolver (client_kwargs, environment, variables, config files,
        EC2 IAM server, in that order). Default: True

    Returns
    -------
    s3_client : boto3.client('s3')
    """
    session = boto3.session.Session()
    if anon:
        s3_client = session.client(
            's3',
            config=Config(signature_version=UNSIGNED)
        )
    else:
        s3_client = session.client('s3')

    return s3_client


def _ls_s3fs(s3_prefix, anon=True):
    """Returns a dict of list of files using s3fs

    The files are divided between subject directories/files and
    non-subject directories/files.

    Parameters
    ----------
    s3_prefix : str
        AWS S3 key for the study or site "directory" that contains all
        of the subjects

    anon : bool
        Whether to use anonymous connection (public buckets only).
        If False, uses the key/secret given, or boto’s credential
        resolver (client_kwargs, environment, variables, config files,
        EC2 IAM server, in that order). Default: True

    Returns
    -------
    subjects : dict
    """
    fs = s3fs.S3FileSystem(anon=anon)
    site_files = fs.ls(s3_prefix, detail=False)

    # Just need BIDSLayout for the `parse_file_entities` method
    dd = tempfile.TemporaryDirectory()
    layout = BIDSLayout(dd.name, validate=False)

    entities = [
        layout.parse_file_entities(f) for f in site_files
    ]

    files = {
        'subjects': [
            f for f, e in zip(site_files, entities)
            if e.get('subject') is not None
        ],
        'other': [
            f for f, e in zip(site_files, entities)
            if e.get('subject') is None
        ]
    }

    return files


def _get_matching_s3_keys(bucket, prefix='', suffix='', anon=True):
    """Generate all the matching keys in an S3 bucket.

    Parameters
    ----------
    bucket : str
        Name of the S3 bucket

    prefix : str, optional
        Only fetch keys that start with this prefix

    suffix : str, optional
        Only fetch keys that end with this suffix

    anon : bool
        Whether to use anonymous connection (public buckets only).
        If False, uses the key/secret given, or boto’s credential
        resolver (client_kwargs, environment, variables, config files,
        EC2 IAM server, in that order). Default: True

    Yields
    ------
    key : list
        S3 keys that match the prefix and suffix
    """
    s3 = get_s3_client(anon=anon)
    kwargs = {'Bucket': bucket, 'MaxKeys': 1000}

    # If the prefix is a single string (not a tuple of strings), we can
    # do the filtering directly in the S3 API.
    if isinstance(prefix, str) and prefix:
        kwargs['Prefix'] = prefix

    while True:
        # The S3 API response is a large blob of metadata.
        # 'Contents' contains information about the listed objects.
        resp = s3.list_objects_v2(**kwargs)

        try:
            contents = resp['Contents']
        except KeyError:
            return

        for obj in contents:
            key = obj['Key']
            if key.startswith(prefix) and key.endswith(suffix):
                yield key

        # The S3 API is paginated, returning up to 1000 keys at a time.
        # Pass the continuation token into the next response, until we
        # reach the final page (when this field is missing).
        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break


def _download_from_s3(fname, bucket, key, overwrite=False, anon=True):
    """Download object from S3 to local file

    Parameters
    ----------
    fname : str
        File path to which to download the object

    bucket : str
        S3 bucket name

    key : str
        S3 key for the object to download

    overwrite : bool
        If True, overwrite file if it already exists.
        If False, skip download and return. Default: False

    anon : bool
        Whether to use anonymous connection (public buckets only).
        If False, uses the key/secret given, or boto’s credential
        resolver (client_kwargs, environment, variables, config files,
        EC2 IAM server, in that order). Default: True
    """
    # Create the directory and file if necessary
    fs = s3fs.S3FileSystem(anon=anon)
    if overwrite or not op.exists(fname):
        Path(op.dirname(fname)).mkdir(parents=True, exist_ok=True)
        fs.get("/".join([bucket, key]), fname)


class S3BIDSSubject:
    """A single study subject hosted on AWS S3"""

    def __init__(self, subject_id, study):
        """Initialize a Subject instance

        Parameters
        ----------
        subject_id : str
            Subject-ID for this subject

        study : AFQ.data.S3BIDSStudy
            The S3BIDSStudy for which this subject was a participant
        """
        logging.getLogger("botocore").setLevel(logging.WARNING)

        if not isinstance(subject_id, str):
            raise TypeError('subject_id must be a string.')

        if not isinstance(study, S3BIDSStudy):
            raise TypeError('study must be an instance of S3BIDSStudy.')

        self._subject_id = subject_id
        self._study = study
        self._get_s3_keys()
        self._files = {'raw': {}, 'derivatives': {}}

    @property
    def subject_id(self):
        """An identifier string for the subject"""
        return self._subject_id

    @property
    def study(self):
        """The study in which this subject participated"""
        return self._study

    @property
    def s3_keys(self):
        """A dict of S3 keys for this subject's data

        The S3 keys are divided between "raw" data and derivatives
        """
        return self._s3_keys

    @property
    def files(self):
        """Local files for this subject's dMRI data

        Before the call to subject.download(), this is None.
        Afterward, the files are stored in a dict with keys
        for each Amazon S3 key and values corresponding to
        the local file.
        """
        return self._files

    def __repr__(self):
        return (f'{type(self).__name__}(subject_id={self.subject_id}, '
                f'study_id={self.study.study_id}')

    def _get_s3_keys(self):
        """Get all required S3 keys for this subject

        Returns
        -------
        s3_keys : dict
            S3 keys organized into "raw" and "derivatives" lists
        """
        prefixes = {
            'raw': '/'.join([self.study.s3_prefix,
                             self.subject_id]).lstrip('/'),
            'derivatives': {
                dt: '/'.join([
                    *dt.split('/')[1:],  # removes bucket name
                    self.subject_id
                ]).lstrip('/') for dt in self.study.derivative_types
            },
        }

        s3_keys = {
            'raw': list(set(_get_matching_s3_keys(
                bucket=self.study.bucket,
                prefix=prefixes['raw'],
                anon=self.study.anon,
            ))),
            'derivatives': {
                dt: list(set(_get_matching_s3_keys(
                    bucket=self.study.bucket,
                    prefix=prefixes['derivatives'][dt],
                    anon=self.study.anon,
                ))) for dt in self.study.derivative_types
            }
        }

        self._s3_keys = s3_keys

    def download(self, directory, include_derivs=False, overwrite=False,
                 suffix=None, pbar=True, pbar_idx=0):
        """Download files from S3

        Parameters
        ----------
        directory : str
            Directory to which to download subject files

        include_derivs : bool or str
            If True, download all derivatives files. If False, do not.
            If a string or sequence of strings is passed, this will
            only download derivatives that match the string(s) (e.g.
            ['dmriprep', 'afq']). Default: False

        overwrite : bool
            If True, overwrite files for each subject. Default: False

        suffix : str
            Suffix, including extension, of file(s) to download.
            Default: None

        pbar : bool
            If True, include download progress bar. Default: True

        pbar_idx : int
            Progress bar index for multithreaded progress bars. Default: 0
        """
        if not isinstance(directory, str):
            raise TypeError('directory must be a string.')

        if not (isinstance(include_derivs, bool)
                or isinstance(include_derivs, str)
                or all(isinstance(s, str) for s in include_derivs)):
            raise TypeError('include_derivs must be a boolean, a '
                            'string, or a sequence of strings.')

        if not isinstance(overwrite, bool):
            raise TypeError('overwrite must be a boolean.')

        if (suffix is not None) and not(isinstance(suffix, str)):
            raise TypeError('suffix must be a string.')

        if not isinstance(pbar, bool):
            raise TypeError('pbar must be a boolean.')

        if not isinstance(pbar_idx, int):
            raise TypeError('pbar_idx must be an integer.')

        def split_key(key):
            if self.study.s3_prefix:
                return key.split(self.study.s3_prefix)[-1]
            else:
                return key

        # Filter out keys that do not end with suffix
        if suffix is not None:
            s3_keys_raw = [
                s3key for s3key in self.s3_keys['raw']
                if s3key.endswith(suffix)
            ]
            s3_keys_deriv = {
                dt: [
                    s3key for s3key in s3keys if s3key.endswith(suffix)
                ] for dt, s3keys in self.s3_keys['derivatives'].items()
            }
        else:
            s3_keys_raw = self.s3_keys['raw']
            s3_keys_deriv = self.s3_keys['derivatives']

        files = {
            'raw': [
                op.abspath(op.join(
                    directory,
                    split_key(key).lstrip('/')
                )) for key in s3_keys_raw
            ],
            'derivatives': {
                dt: [
                    op.abspath(op.join(
                        directory,
                        split_key(s3key).lstrip('/')
                    )) for s3key in s3keys
                ] for dt, s3keys in s3_keys_deriv.items()
            }
        }

        raw_zip = list(zip(s3_keys_raw, files['raw']))

        # Populate files parameter
        self._files["raw"].update({k: f for k, f in raw_zip})

        # Generate list of (key, file) tuples
        download_pairs = [(k, f) for k, f in raw_zip]

        deriv_zips = {
            dt: list(zip(
                s3keys, files['derivatives'][dt]
            )) for dt, s3keys in s3_keys_deriv.items()
        }

        deriv_pairs = []
        for dt in files['derivatives'].keys():
            if include_derivs is True:
                # In this case, include all derivatives files
                deriv_pairs += [(k, f) for k, f in deriv_zips[dt]]
                self._files['derivatives'][dt] = {
                    k: f for k, f in deriv_zips[dt]
                }
            elif include_derivs is False:
                pass
            elif (isinstance(include_derivs, str)
                  # In this case, filter only derivatives S3 keys that
                  # include the `include_derivs` string as a substring
                  and include_derivs in dt):
                deriv_pairs += [(k, f) for k, f in deriv_zips[dt]]
                self._files['derivatives'][dt] = {
                    k: f for k, f in deriv_zips[dt]
                }
            elif (all(isinstance(s, str) for s in include_derivs)
                  and any([deriv in dt for deriv in include_derivs])):
                # In this case, filter only derivatives S3 keys that
                # include any of the `include_derivs` strings as a
                # substring
                deriv_pairs += [(k, f) for k, f in deriv_zips[dt]]
                self._files['derivatives'][dt] = {
                    k: f for k, f in deriv_zips[dt]
                }

        if include_derivs is not False:
            download_pairs += deriv_pairs

        # Now iterate through the list and download each item
        if pbar:
            progress = tqdm(desc=f'Download {self.subject_id}',
                            position=pbar_idx,
                            total=len(download_pairs) + 1)

        for (key, fname) in download_pairs:
            _download_from_s3(fname=fname,
                              bucket=self.study.bucket,
                              key=key,
                              overwrite=overwrite,
                              anon=self.study.anon)

            if pbar:
                progress.update()

        if pbar:
            progress.update()
            progress.close()


class HBNSubject(S3BIDSSubject):
    """A subject in the HBN study

    See Also
    --------
    AFQ.data.S3BIDSSubject
    """

    def __init__(self, subject_id, study, site=None):
        """Initialize a Subject instance

        Parameters
        ----------
        subject_id : str
            Subject-ID for this subject

        study : AFQ.data.S3BIDSStudy
            The S3BIDSStudy for which this subject was a participant

        site : str, optional
            Site-ID for the site from which this subject's data was collected
        """
        if not (site is None or isinstance(site, str)):
            raise TypeError('site must be a string or None.')

        self._site = site

        super().__init__(
            subject_id=subject_id,
            study=study
        )

    @property
    def site(self):
        """The site at which this subject was a participant"""
        return self._site

    def __repr__(self):
        return (f'{type(self).__name__}(subject_id={self.subject_id}, '
                f'study_id={self.study.study_id}, site={self.site}')

    def _get_s3_keys(self):
        """Get all required S3 keys for this subject

        Returns
        -------
        s3_keys : dict
            S3 keys organized into "raw" and "derivatives" lists
        """
        prefixes = {
            'raw': '/'.join([self.study.s3_prefix,
                             self.subject_id]).lstrip('/'),
            'derivatives': '/'.join([
                self.study.s3_prefix,
                'derivatives',
                self.subject_id
            ]).lstrip('/')
        }

        s3_keys = {
            datatype: list(set(_get_matching_s3_keys(
                bucket=self.study.bucket,
                prefix=prefix,
                anon=self.study.anon,
            ))) for datatype, prefix in prefixes.items()
        }

        def get_deriv_type(s3_key):
            after_sub = s3_key.split('/' + self.subject_id + '/')[-1]
            deriv_type = after_sub.split('/')[0]
            return deriv_type

        deriv_keys = {
            dt: [
                s3key for s3key in s3_keys['derivatives']
                if dt == get_deriv_type(s3key)
            ] for dt in self.study.derivative_types
        }

        s3_keys['derivatives'] = deriv_keys
        self._s3_keys = s3_keys


class S3BIDSStudy:
    """A BIDS-compliant study hosted on AWS S3"""

    def __init__(self, study_id, bucket, s3_prefix='', subjects=None,
                 anon=True, use_participants_tsv=False, random_seed=None,
                 _subject_class=S3BIDSSubject):
        """Initialize an S3BIDSStudy instance

        Parameters
        ----------
        study_id : str
            An identifier string for the study

        bucket : str
            The S3 bucket that contains the study data

        s3_prefix : str, optional
            The S3 prefix common to all of the study objects on S3.
            Default: the empty string, which indicates that the study
            is at the top level of the bucket.

        subjects : str, sequence(str), int, or None, optional
            If int, retrieve S3 keys for the first `subjects` subjects.
            If "all", retrieve all subjects. If str or sequence of
            strings, retrieve S3 keys for the specified subjects. If sequence
            of ints, then for each int n retrieve S3 keys for the nth subject.
            If None, retrieve S3 keys for the first subject. Default: None

        anon : bool, optional
            Whether to use anonymous connection (public buckets only).
            If False, uses the key/secret given, or boto’s credential
            resolver (client_kwargs, environment, variables, config
            files, EC2 IAM server, in that order). Default: True

        use_participants_tsv : bool, optional
            If True, use the particpants tsv files to retrieve subject
            identifiers. This is faster but may not catch all subjects.
            Sometimes the tsv files are outdated. Default: False

        random_seed : int or None, optional
            Random seed for selection of subjects if `subjects` is an
            integer. Use the same random seed for reproducibility.
            Default: None

        _subject_class : object, optional
            The subject class to be used for this study. This parameter
            has a leading underscore because you probably don't want
            to change it. If you do change it, you must provide a
            class that quacks like AFQ.data.S3BIDSSubject. Default:
            S3BIDSSubject

        Examples
        --------
        Access data stored in a bucket using credentials:
        >>> study = S3BIDSStudy('studyname',
        ...                     'bucketname',
        ...                     '/path/to/dataset/',
        ...                     anon=False)

        Access data stored in a publicly accessible bucket:
        >>> study = S3BIDSStudy('hbn',
        ...    'fcp-indi',
        ...    'data/Projects/HBN/BIDS_curated/derivatives/qsiprep/')

        """
        logging.getLogger("botocore").setLevel(logging.WARNING)

        if not isinstance(study_id, str):
            raise TypeError('`study_id` must be a string.')

        if not isinstance(bucket, str):
            raise TypeError('`bucket` must be a string.')

        if not isinstance(s3_prefix, str):
            raise TypeError('`s3_prefix` must be a string.')

        if s3_prefix == '/':
            raise ValueError("If the study is at the top level "
                             "of the s3 bucket, please pass the "
                             "empty string as the s3 prefix"
                             "(the default value)")
        if not (subjects is None
                or isinstance(subjects, int)
                or isinstance(subjects, str)
                or all(isinstance(s, str) for s in subjects)
                or all(isinstance(s, int) for s in subjects)):
            raise TypeError('`subjects` must be an int, string, '
                            'sequence of strings, or a sequence of ints.')

        if not isinstance(anon, bool):
            raise TypeError('`anon` must be of type bool.')

        if isinstance(subjects, int) and subjects < 1:
            raise ValueError('If `subjects` is an int, it must be '
                             'greater than 0.')

        if not isinstance(use_participants_tsv, bool):
            raise TypeError('`use_participants_tsv` must be boolean.')

        if not (random_seed is None or isinstance(random_seed, int)):
            raise TypeError("`random_seed` must be an integer.")

        self._study_id = study_id
        self._bucket = bucket
        self._s3_prefix = s3_prefix
        self._use_participants_tsv = use_participants_tsv
        self._random_seed = random_seed
        self._anon = anon
        self._subject_class = _subject_class
        self._local_directories = []

        # Get a list of all subjects in the study
        self._all_subjects = self._list_all_subjects()
        self._derivative_types = self._get_derivative_types()
        self._non_subject_s3_keys = self._get_non_subject_s3_keys()

        # Convert `subjects` into a sequence of subjectID strings
        if subjects is None or isinstance(subjects, int) \
                or (isinstance(subjects, list)
                    and isinstance(subjects[0], int)):
            # if subjects is an int, get that many random subjects
            prng = np.random.RandomState(random_seed)
            randomized_subjects = sorted(self._all_subjects.copy())
            prng.shuffle(randomized_subjects)

            if subjects is None:
                subjects = randomized_subjects[0]
            elif isinstance(subjects, int):
                subjects = randomized_subjects[:subjects]
            else:
                subjects = [randomized_subjects[i] for i in subjects]

            if isinstance(subjects, str):
                subjects = [subjects]
        elif subjects == 'all':
            # if "all," retrieve all subjects
            subjects = sorted(self._all_subjects)
        elif isinstance(subjects, str):
            # if a string, just get that one subject
            subjects = [subjects]
        # The last case for subjects is what we want. No transformation needed.

        if not set(subjects) <= set(self._all_subjects):
            raise ValueError(
                f'The following subjects could not be found in the study: '
                f'{set(subjects) - set(self._all_subjects)}'
            )

        subs = [
            delayed(self._get_subject)(s) for s in set(subjects)
        ]

        print('Retrieving subject S3 keys')
        with ProgressBar():
            subjects = list(compute(*subs, scheduler='threads'))

        self._subjects = subjects

    @property
    def study_id(self):
        """An identifier string for the study"""
        return self._study_id

    @property
    def bucket(self):
        """The S3 bucket that contains the study data"""
        return self._bucket

    @property
    def s3_prefix(self):
        """The S3 prefix common to all of the study objects on S3"""
        return self._s3_prefix

    @property
    def subjects(self):
        """A list of Subject instances for each requested subject"""
        return self._subjects

    @property
    def anon(self):
        """Is this study using an anonymous S3 connection?"""
        return self._anon

    @property
    def derivative_types(self):
        """A list of derivative pipelines available in this study"""
        return self._derivative_types

    @property
    def non_sub_s3_keys(self):
        """A dict of S3 keys that are not in subject directories"""
        return self._non_subject_s3_keys

    @property
    def local_directories(self):
        """A list of local directories where this study has been downloaded"""
        return self._local_directories

    @property
    def use_participants_tsv(self):
        """Did we use a participants.tsv file to populate the list of
        study subjects."""
        return self._use_participants_tsv

    @property
    def random_seed(self):
        """The random seed used to retrieve study subjects"""
        return self._random_seed

    def __repr__(self):
        return (f'{type(self).__name__}(study_id={self.study_id}, '
                f'bucket={self.bucket}, s3_prefix={self.s3_prefix})')

    def _get_subject(self, subject_id):
        """Return a Subject instance from a subject-ID"""
        return self._subject_class(subject_id=subject_id,
                                   study=self)

    def _get_derivative_types(self):
        """Return a list of available derivatives pipelines

        Returns
        -------
        list
            list of available derivatives pipelines
        """
        s3_prefix = '/'.join([self.bucket, self.s3_prefix]).rstrip("/")
        nonsub_keys = _ls_s3fs(s3_prefix=s3_prefix,
                               anon=self.anon)['other']
        derivatives_prefix = '/'.join([s3_prefix, 'derivatives'])
        if derivatives_prefix in nonsub_keys:
            return _ls_s3fs(
                s3_prefix=derivatives_prefix,
                anon=self.anon
            )['other']
        else:
            return []

    def _get_non_subject_s3_keys(self):
        """Return a list of 'non-subject' files

        In this context, a 'non-subject' file is any file
        or directory that is not a subject ID folder

        Returns
        -------
        dict
            dict with keys 'raw' and 'derivatives' and whose values
            are lists of S3 keys for non-subject files
        """
        non_sub_s3_keys = {}

        s3_prefix = '/'.join([self.bucket, self.s3_prefix]).rstrip("/")

        nonsub_keys = _ls_s3fs(s3_prefix=s3_prefix,
                               anon=self.anon)['other']
        nonsub_keys = [k for k in nonsub_keys
                       if not k.endswith('derivatives')]

        nonsub_deriv_keys = []
        for dt in self.derivative_types:
            nonsub_deriv_keys.append(_ls_s3fs(
                s3_prefix=dt,
                anon=self.anon
            )['other'])

        non_sub_s3_keys = {
            'raw': nonsub_keys,
            'derivatives': nonsub_deriv_keys,
        }

        return non_sub_s3_keys

    def _list_all_subjects(self):
        """Return list of subjects

        Returns
        -------
        list
            list of participant_ids
        """
        if self._use_participants_tsv:
            tsv_key = "/".join([self.s3_prefix,
                                "participants.tsv"]).lstrip("/")
            s3 = get_s3_client(anon=self.anon)

            def get_subs_from_tsv_key(s3_key):
                response = s3.get_object(
                    Bucket=self.bucket,
                    Key=s3_key
                )

                return set(pd.read_csv(
                    response.get('Body')
                ).participant_id.values)

            subject_set = get_subs_from_tsv_key(tsv_key)
            subjects = list(subject_set)
        else:
            s3_prefix = '/'.join([self.bucket, self.s3_prefix]).rstrip("/")
            sub_keys = _ls_s3fs(s3_prefix=s3_prefix,
                                anon=self.anon)['subjects']

            # Just need BIDSLayout for the `parse_file_entities`
            dd = tempfile.TemporaryDirectory()
            layout = BIDSLayout(dd.name, validate=False)
            subjects = []
            for key in sub_keys:
                entities = layout.parse_file_entities(key)
                subjects.append('sub-' + entities.get('subject'))

        return list(set(subjects))

    def _download_non_sub_keys(self, directory,
                               select=("dataset_description.json",),
                               filenames=None):
        fs = s3fs.S3FileSystem(anon=self.anon)
        if filenames is None:
            filenames = self.non_sub_s3_keys['raw']
        for fn in filenames:
            if select == "all" or any([s in fn for s in select]):
                Path(directory).mkdir(parents=True, exist_ok=True)
                fs.get(fn, op.join(directory, op.basename(fn)))

    def _download_derivative_descriptions(self, include_derivs, directory):
        for derivative in self.derivative_types:
            if include_derivs is True \
                or (isinstance(include_derivs, str)
                    and include_derivs == op.basename(derivative)) \
                or (isinstance(include_derivs, list)
                    and all(isinstance(s, str) for s in include_derivs)
                    and any([deriv in derivative for
                             deriv in include_derivs])):
                filenames = \
                    _ls_s3fs(s3_prefix=derivative, anon=self.anon)['other']
                deriv_directory = op.join(
                    directory, *derivative.split('/')[-2:])
                self._download_non_sub_keys(
                    deriv_directory,
                    select=("dataset_description.json",),
                    filenames=filenames)

    def download(self, directory,
                 include_modality_agnostic=("dataset_description.json",),
                 include_derivs=False,
                 include_derivs_dataset_description=True,
                 suffix=None,
                 overwrite=False, pbar=True):
        """Download files for each subject in the study

        Parameters
        ----------
        directory : str
            Directory to which to download subject files

        include_modality_agnostic : bool, "all" or any subset of [
                "dataset_description.json", "CHANGES", "README", "LICENSE"]
            If True or "all", download all keys in self.non_sub_s3_keys
            also. If a subset of ["dataset_description.json", "CHANGES",
            "README", "LICENSE"], download only those files. This is
            useful if the non_sub_s3_keys contain files common to all
            subjects that should be inherited.
            Default: ("dataset_description.json",)

        include_derivs : bool or str
            If True, download all derivatives files. If False, do not.
            If a string or sequence of strings is passed, this will
            only download derivatives that match the string(s) (e.g.
            ["dmriprep", "afq"]). Default: False

        include_derivs_dataset_description : bool
            Used only if include_derivs is not False. If True,
            dataset_description.json downloaded for each derivative.

        suffix : str
            Suffix, including extension, of file(s) to download.
            Default: None

        overwrite : bool
            If True, overwrite files for each subject. Default: False

        pbar : bool
            If True, include progress bar. Default: True

        See Also
        --------
        AFQ.data.S3BIDSSubject.download
        """
        self._local_directories.append(directory)
        self._local_directories = list(set(self._local_directories))

        if include_modality_agnostic is True\
                or include_modality_agnostic == "all":
            self._download_non_sub_keys(directory, select="all")
        elif include_modality_agnostic is not False:
            # Subset selection
            valid_set = {"dataset_description.json",
                         "CHANGES",
                         "README",
                         "LICENSE"}
            if not set(include_modality_agnostic) <= valid_set:
                raise ValueError(
                    "include_modality_agnostic must be either"
                    " a boolean, 'all', "
                    "or a subset of {valid_set}".format(valid_set=valid_set)
                )

            self._download_non_sub_keys(
                directory, select=include_modality_agnostic)

        # download dataset_description.json for derivatives
        if (include_derivs is not False) \
                and include_derivs_dataset_description:
            self._download_derivative_descriptions(
                include_derivs, directory)

        results = [delayed(sub.download)(
            directory=directory,
            include_derivs=include_derivs,
            suffix=suffix,
            overwrite=overwrite,
            pbar=pbar,
            pbar_idx=idx,
        ) for idx, sub in enumerate(self.subjects)]

        compute(*results, scheduler='threads')


class HBNSite(S3BIDSStudy):
    """An HBN study site

    See Also
    --------
    AFQ.data.S3BIDSStudy
    """

    def __init__(self, site, study_id='HBN', bucket='fcp-indi',
                 s3_prefix='data/Projects/HBN/MRI',
                 subjects=None, use_participants_tsv=False,
                 random_seed=None):
        """Initialize the HBN site

        Parameters
        ----------
        site : ["Site-SI", "Site-RU", "Site-CBIC", "Site-CUNY"]
            The HBN site

        study_id : str
            An identifier string for the site

        bucket : str
            The S3 bucket that contains the study data

        s3_prefix : str
            The S3 prefix common to all of the study objects on S3

        subjects : str, sequence(str), int, or None
            If int, retrieve S3 keys for the first `subjects` subjects.
            If "all", retrieve all subjects. If str or sequence of
            strings, retrieve S3 keys for the specified subjects. If
            None, retrieve S3 keys for the first subject. Default: None

        use_participants_tsv : bool
            If True, use the particpants tsv files to retrieve subject
            identifiers. This is faster but may not catch all subjects.
            Sometimes the tsv files are outdated. Default: False

        random_seed : int or None
            Random seed for selection of subjects if `subjects` is an
            integer. Use the same random seed for reproducibility.
            Default: None
        """
        valid_sites = ["Site-SI", "Site-RU", "Site-CBIC", "Site-CUNY"]
        if site not in valid_sites:
            raise ValueError(
                "site must be one of {}.".format(valid_sites)
            )

        self._site = site

        super().__init__(
            study_id=study_id,
            bucket=bucket,
            s3_prefix='/'.join([s3_prefix, site]),
            subjects=subjects,
            use_participants_tsv=use_participants_tsv,
            random_seed=random_seed,
            _subject_class=HBNSubject
        )

    @property
    def site(self):
        """The HBN site"""
        return self._site

    def _get_subject(self, subject_id):
        """Return a Subject instance from a subject-ID"""
        return self._subject_class(subject_id=subject_id,
                                   study=self,
                                   site=self.site)

    def _get_derivative_types(self):
        """Return a list of available derivatives pipelines

        The HBN dataset is not BIDS compliant so to go a list
        of available derivatives, we must peak inside every
        directory in `derivatives/sub-XXXX/`

        Returns
        -------
        list
            list of available derivatives pipelines
        """
        s3_prefix = '/'.join([self.bucket, self.s3_prefix]).rstrip("/")
        nonsub_keys = _ls_s3fs(s3_prefix=s3_prefix,
                               anon=self.anon)['other']
        derivatives_prefix = '/'.join([s3_prefix, 'derivatives'])

        if any([derivatives_prefix in key for key in nonsub_keys]):
            deriv_subs = _ls_s3fs(
                s3_prefix=derivatives_prefix,
                anon=self.anon
            )['subjects']

            deriv_types = []
            for sub_key in deriv_subs:
                deriv_types += [
                    s.split(sub_key)[-1].lstrip('/')
                    for s in _ls_s3fs(
                        s3_prefix=sub_key,
                        anon=self.anon
                    )['subjects']
                ]

            return list(set(deriv_types))
        else:
            return []

    def _get_non_subject_s3_keys(self):
        """Return a list of 'non-subject' files

        In this context, a 'non-subject' file is any file
        or directory that is not a subject ID folder. This method
        is different from AFQ.data.S3BIDSStudy because the HBN
        dataset is not BIDS compliant

        Returns
        -------
        dict
            dict with keys 'raw' and 'derivatives' and whose values
            are lists of S3 keys for non-subject files

        See Also
        --------
        AFQ.data.S3BIDSStudy._get_non_subject_s3_keys
        """
        non_sub_s3_keys = {}

        s3_prefix = '/'.join([self.bucket, self.s3_prefix]).rstrip("/")

        nonsub_keys = _ls_s3fs(s3_prefix=s3_prefix,
                               anon=self.anon)['other']
        nonsub_keys = [k for k in nonsub_keys
                       if not k.endswith('derivatives')]

        nonsub_deriv_keys = _ls_s3fs(
            s3_prefix='/'.join([
                self.bucket,
                self.s3_prefix,
                'derivatives'
            ]),
            anon=self.anon
        )['other']

        non_sub_s3_keys = {
            'raw': nonsub_keys,
            'derivatives': nonsub_deriv_keys,
        }

        return non_sub_s3_keys

    def download(self, directory, include_modality_agnostic=False,
                 include_derivs=False, overwrite=False, pbar=True):
        """Download files for each subject in the study

        Parameters
        ----------
        directory : str
            Directory to which to download subject files

        include_modality_agnostic : bool, "all" or any subset of [
                "dataset_description.json", "CHANGES", "README", "LICENSE"]
            If True or "all", download all keys in self.non_sub_s3_keys
            also. If a subset of ["dataset_description.json", "CHANGES",
            "README", "LICENSE"], download only those files. This is
            useful if the non_sub_s3_keys contain files common to all
            subjects that should be inherited. Default: False

        include_derivs : bool or str
            If True, download all derivatives files. If False, do not.
            If a string or sequence of strings is passed, this will
            only download derivatives that match the string(s) (e.g.
            ["dmriprep", "afq"]). Default: False

        overwrite : bool
            If True, overwrite files for each subject. Default: False

        pbar : bool
            If True, include progress bar. Default: True

        See Also
        --------
        AFQ.data.S3BIDSSubject.download
        """
        super().download(
            directory=directory,
            include_modality_agnostic=include_modality_agnostic,
            include_derivs=include_derivs,
            overwrite=overwrite,
            pbar=pbar
        )

        to_bids_description(
            directory,
            **{"BIDSVersion": "1.0.0",
               "Name": "HBN Study, " + self.site,
               "DatasetType": "raw",
               "Subjects": [s.subject_id for s in self.subjects]})


# +--------------------------------------------------+
# | End S3BIDSStudy classes and supporting functions |
# +--------------------------------------------------+

def s3fs_nifti_write(img, fname, fs=None):
    """
    Write a nifti file straight to S3

    Paramters
    ---------
    img : nib.Nifti1Image class instance
        The image containing data to be written into S3
    fname : string
        Full path (including bucket name and extension) to the S3 location
        where the file is to be saved.
    fs : an s3fs.S3FileSystem class instance, optional
        A file-system to refer to. Default to create a new file-system
    """
    if fs is None:
        fs = s3fs.S3FileSystem()

    bio = BytesIO()
    file_map = img.make_file_map({'image': bio, 'header': bio})
    img.to_file_map(file_map)
    data = gzip.compress(bio.getvalue())
    with fs.open(fname, 'wb') as ff:
        ff.write(data)


def s3fs_nifti_read(fname, fs=None, anon=False):
    """
    Lazily reads a nifti image from S3.

    Paramters
    ---------
    fname : string
        Full path (including bucket name and extension) to the S3 location
        of the file to be read.
    fs : an s3fs.S3FileSystem class instance, optional
        A file-system to refer to. Default to create a new file-system.
    anon : bool
        Whether to use anonymous connection (public buckets only).
        If False, uses the key/secret given, or boto’s credential
        resolver (client_kwargs, environment, variables, config files,
        EC2 IAM server, in that order). Default: True

    Returns
    -------
    nib.Nifti1Image class instance

    Notes
    -----
    Because the image is lazily loaded, data stored in the file
    is not transferred until `get_fdata` is called.

    """
    if fs is None:
        fs = s3fs.S3FileSystem(anon=anon)
    with fs.open(fname) as ff:
        zz = gzip.open(ff)
        rr = zz.read()
        bb = BytesIO(rr)
        fh = nib.FileHolder(fileobj=bb)
        img = nib.Nifti1Image.from_file_map({'header': fh, 'image': fh})
    return img


def write_json(fname, data):
    """
    Write data to JSON file.

    Parameters
    ----------
    fname : str
        Full path to the file to write.

    data : dict
        A dict containing the data to write.

    Returns
    -------
    None
    """
    with open(fname, 'w') as ff:
        json.dump(data, ff, default=lambda obj: "Not Serializable")


def read_json(fname):
    """
    Read data from a JSON file.

    Parameters
    ----------
    fname : str
        Full path to the data-containing file

    Returns
    -------
    dict
    """
    with open(fname, 'r') as ff:
        out = json.load(ff)
    return out


def s3fs_json_read(fname, fs=None, anon=False):
    """
    Reads json directly from S3

    Paramters
    ---------
    fname : str
        Full path (including bucket name and extension) to the file on S3.
    fs : an s3fs.S3FileSystem class instance, optional
        A file-system to refer to. Default to create a new file-system.
    anon : bool
        Whether to use anonymous connection (public buckets only).
        If False, uses the key/secret given, or boto’s credential
        resolver (client_kwargs, environment, variables, config files,
        EC2 IAM server, in that order). Default: True
    """
    if fs is None:
        fs = s3fs.S3FileSystem(anon=anon)
    with fs.open(fname) as ff:
        data = json.load(ff)
    return data


def s3fs_json_write(data, fname, fs=None):
    """
    Writes json from a dict directly into S3

    Parameters
    ----------
    data : dict
        The json to be written out
    fname : str
        Full path (including bucket name and extension) to the file to
        be written out on S3
    fs : an s3fs.S3FileSystem class instance, optional
        A file-system to refer to. Default to create a new file-system.
    """
    if fs is None:
        fs = s3fs.S3FileSystem()
    with fs.open(fname, 'w') as ff:
        json.dump(data, ff)
