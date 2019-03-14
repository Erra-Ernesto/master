# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import tarfile
from zipfile import ZipFile

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

import tensorflow as tf

_ENIT_TRAIN_DATASETS = [
    [
        "http://download1193.mediafire.com/0ljw8tp9q76g/1pwrjzcmtok4zwo/ParaCrawl.tar",  # ParaCrawl
        ("ParaCrawl/ParaCrawl.en-it.en",
         "ParaCrawl/ParaCrawl.en-it.it")
    ],
    [
        "http://download1193.mediafire.com/0ljw8tp9q76g/1pwrjzcmtok4zwo/Books.tar",
        ("Books.txt/Books.en-it.en",
         "Books.txt/Books.en-it.it")
    ]
    #[
    #    "http://download1193.mediafire.com/0ljw8tp9q76g/1pwrjzcmtok4zwo/Europarl_v7.tar",  # Europarl_v7
    #    ("Europarl_v7/Europarl.en-it.en",
    #     "Europarl_v7/Europarl.en-it.it")
    #],
    #[
    #    "http://fuffa.com/EUbookshop.tar",  # EUbookshop
    #    ("EUbookshop/EUbookshop.en-it.en",
    #     "EUbookshop/EUbookshop.en-it.it")
    #]
]

_ENIT_TEST_DATASETS = [
    [
        "http://download1869.mediafire.com/4728hs18i7kg/yibn5iibdy4t6yk/News-Commentary.tar",  # Newscommentary11
        ("News-Commentary/News-Commentary.en-it.en",
         "News-Commentary/News-Commentary.en-it.it")
    ]
#    [
#        "http://dw.convertfiles.com/files/0237482001525345058/OpenSubtitles.tar",
#        ("OpenSubtitles/OpenSubtitles.en-it.en",
#         "OpenSubtitles/OpenSubtitles.en-it.it")
#    ]
]


def convert(ifn, ofn):
    with ZipFile(ifn) as zipf:
        with tarfile.open(ofn, 'w:bz2') as tarf:
            for zip_info in zipf.infolist():
                # print zip_info.filename, zip_info.file_size
                tar_info = tarfile.TarInfo(name=zip_info.filename)
                tar_info.size = zip_info.file_size
                tar_info.mtime = time.mktime(list(zip_info.date_time) +
                                             [-1, -1, -1])
                tarf.addfile(
                    tarinfo=tar_info,
                    fileobj=zipf.open(zip_info.filename)
                )
            return tarf


def _get_wmt_enit_bpe_dataset(directory, filename):
    """Extract the WMT en-it corpus `filename` to directory unless it's there."""
    train_path = os.path.join(directory, filename)
    if not (tf.gfile.Exists(train_path + ".it") and
            tf.gfile.Exists(train_path + ".en")):
        url = ("https://drive.google.com/open?id=1F3apMpe1lijbUzZfMNPBJlvURgV3Sx2t")
        corpus_file = generator_utils.maybe_download_from_drive(
            directory, "News-Commentary-enit.tar", url)

        # convert zip to tar

        with tarfile.open(corpus_file, "r:gz") as corpus_tar:
            corpus_tar.extractall(directory)
    return train_path


@registry.register_problem
class TranslateEnitWmtBpe32k(translate.TranslateProblem):
    """Problem spec for WMT En-It translation, BPE version."""

    @property
    def approx_vocab_size(self):
        return 32000

    @property
    def vocab_filename(self):
        return "vocab.bpe.%d" % self.approx_vocab_size

    def get_or_create_vocab(self, data_dir, tmp_dir, force_get=False):
        vocab_filename = os.path.join(data_dir, self.vocab_filename)
        if not tf.gfile.Exists(vocab_filename) and force_get:
            raise ValueError("Vocab %s not found" % vocab_filename)
        return text_encoder.TokenTextEncoder(vocab_filename, replace_oov="UNK")

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        """Instance of token generator for the WMT en->de task, training set."""
        train = dataset_split == problem.DatasetSplit.TRAIN
        dataset_path = ("train.tok.clean.bpe.32000"
                        if train else "newstest2013.tok.bpe.32000")  # da controllare
        train_path = _get_wmt_enit_bpe_dataset(tmp_dir, dataset_path)

        # Vocab
        token_path = os.path.join(data_dir, self.vocab_filename)
        if not tf.gfile.Exists(token_path):
            token_tmp_path = os.path.join(tmp_dir, self.vocab_filename)
            tf.gfile.Copy(token_tmp_path, token_path)
            with tf.gfile.GFile(token_path, mode="r") as f:
                vocab_data = "<pad>\n<EOS>\n" + f.read() + "UNK\n"
            with tf.gfile.GFile(token_path, mode="w") as f:
                f.write(vocab_data)

        return text_problems.text2text_txt_iterator(train_path + ".en",
                                                    train_path + ".it")


@registry.register_problem
class TranslateEnitWmt8k(translate.TranslateProblem):
    """Problem spec for WMT En-De translation."""

    @property
    def approx_vocab_size(self):
        return 2 ** 13  # 8192

    @property
    def vocab_filename(self):
        return "vocab.enit.%d" % self.approx_vocab_size

    def source_data_files(self, dataset_split):
        train = dataset_split == problem.DatasetSplit.TRAIN
        return _ENIT_TRAIN_DATASETS if train else _ENIT_TEST_DATASETS


@registry.register_problem
class TranslateEnitWmt32k(TranslateEnitWmt8k):

    @property
    def approx_vocab_size(self):
        return 2 ** 15  # 32768


@registry.register_problem
class TranslateEnitWmt32kPacked(TranslateEnitWmt32k):

    @property
    def packed_length(self):
        return 256


@registry.register_problem
class TranslateEnitWmt8kPacked(TranslateEnitWmt8k):

    @property
    def packed_length(self):
        return 256


@registry.register_problem
class TranslateEnitWmtCharacters(translate.TranslateProblem):
    """Problem spec for WMT En-It translation."""

    @property
    def vocab_type(self):
        return text_problems.VocabType.CHARACTER

