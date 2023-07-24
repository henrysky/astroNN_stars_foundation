import copy

import torch
import numpy as np
from typing import List

from utils.data_utils import shuffle_row, random_choice

rng = np.random.default_rng()


def robust_mean_squared_error(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    variance: torch.Tensor,
    labels_err: torch.Tensor,
) -> torch.Tensor:
    # Neural Net is predicting log(var), so take exp, takes account the target variance, and take log back
    total_var = torch.exp(variance) + torch.square(labels_err)
    wrapper_output = 0.5 * (
        (torch.square(y_true - y_pred) / total_var) + torch.log(total_var)
    )

    losses = wrapper_output.sum() / y_true.shape[0]
    return losses


def mean_squared_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    losses = (torch.square(y_true - y_pred)).sum() / y_true.shape[0]
    return losses


class TrainingGenerator(torch.utils.data.Dataset):
    def __init__(
        self,
        batch_size: int,
        data: dict,
        outputs_padding: int=0,
        possible_output_tokens: List[int]=None,
        input_length: int=None,
        shuffle: bool=True,
        aggregate_nans: bool=True,
        factory_kwargs: dict={"device": "cpu", "dtype": torch.float32},
    ):
        """
        Parameters
        ----------
        batch_size : int
            batch size
        data : dict
            data dictionary that contains the following keys: ["input", "input_token", "output", "output_err"]
        outputs_padding : int, optional
            additional padding for output, by default 0
        possible_output_tokens : np.ndarray, optional
            possible output tokens, by default None
        input_length : int, optional
            input length, by default None
        shuffle : bool, optional
            shuffle data, by default True
        aggregate_nans : bool, optional
            aggregate nans of every rows to the end of those rows, by default True
        """
        self.factory_kwargs = factory_kwargs
        self.input = copy.deepcopy(data["input"])
        self.input_idx = copy.deepcopy(data["input_token"])
        self.output = data["output"]
        self.output_err = copy.deepcopy(data["output_err"])
        self.outputs_padding = outputs_padding
        self.data_length = len(self.input)
        self.data_width = self.input.shape[1]
        self.shuffle = shuffle  # shuffle row ordering, star level column ordering shuffle is mandatory
        self.aggregate_nans = aggregate_nans

        # handle possible output tokens for star-by-star basis
        self.possible_output_tokens = possible_output_tokens
        prob_matrix = np.tile(
            np.ones_like(possible_output_tokens, dtype=float), (self.data_length, 1)
        )
        bad_idx = (
            self.input_idx[
                np.arange(self.data_length),
                np.expand_dims(self.possible_output_tokens - 1, -1),
            ]
            == 0
        ).T
        # only need to do this once, very time consuming
        if aggregate_nans:  # aggregate nans to the end of each row
            # partialsort_idx = np.argpartition(self.input_idx, np.sum(self.input_idx == 0, axis=1), axis=1)
            partialsort_idx = np.argsort(self.input_idx == 0, axis=1, kind="mergesort")
            self.input = np.take_along_axis(self.input, partialsort_idx, axis=1)
            self.input_idx = np.take_along_axis(self.input_idx, partialsort_idx, axis=1)
            self.first_n_shuffle = self.data_width - np.sum(self.input_idx == 0, axis=1)

        else:
            self.first_n_shuffle = None

        # ================ temperory ================
        # prob_matrix[24, :] = 30
        # prob_matrix[25, :] = 30
        # prob_matrix[26, :] = 30
        # ================ temperory ================

        prob_matrix[
            bad_idx
        ] = 0.0  # don't sample those token which are missing (i.e. padding)
        self.output_prob_matrix = prob_matrix

        self.batch_size = batch_size
        self.steps_per_epoch = self.data_length // self.batch_size

        # placeholder to epoch level data
        self.epoch_input = None
        self.epoch_input_idx = None
        self.epoch_output = None
        self.epoch_output_idx = None

        self.idx_list = np.arange(self.data_length)

        self.input_length = input_length
        if self.input_length is None:
            self.input_length = data["input"].shape[1]

        # we put every preparation in on_epoch_end()
        self.on_epoch_end()

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for i in range(len(self)):
            idx_list_temp = self.idx_list[
                i * self.batch_size : (i + 1) * self.batch_size
            ]
            yield (
                self.epoch_input[idx_list_temp],
                self.epoch_input_idx[idx_list_temp],
                self.epoch_output_idx[idx_list_temp],
                self.epoch_output[idx_list_temp],
                self.epoch_output_err[idx_list_temp],
            )

    def __len__(self):
        return self.steps_per_epoch

    def on_epoch_end(self):
        """
        Major functionality is to prepare the data for the next epoch
        """
        # shuffle the row ordering list when an epoch ends to prepare for the next epoch
        if self.shuffle:
            rng.shuffle(self.idx_list)

        self.epoch_input = copy.deepcopy(self.input)
        self.epoch_input_idx = copy.deepcopy(self.input_idx)
        shuffle_row(
            [self.epoch_input, self.epoch_input_idx], first_n=self.first_n_shuffle
        )

        # crop
        self.epoch_input = self.epoch_input[:, : self.input_length]
        self.epoch_input_idx = self.epoch_input_idx[:, : self.input_length]
        # add random padding
        if self.outputs_padding != 0:
            padding_length = rng.choice(
                np.arange(0, self.outputs_padding), size=self.data_length
            )
            for idx, pad in enumerate(padding_length):
                if pad != 0:  # 0 means using all, so dont mask
                    self.epoch_input[idx, -pad:] = 0.0
                    self.epoch_input_idx[idx, -pad:] = 0
        # choose one depending on output prob matrix
        output_idx = random_choice(
            np.tile(self.possible_output_tokens, (self.data_length, 1)),
            self.output_prob_matrix,
        )

        self.epoch_input = torch.atleast_3d(
            torch.tensor(self.epoch_input, **self.factory_kwargs)
        )
        self.epoch_input_idx = torch.tensor(
            self.epoch_input_idx,
            device=self.factory_kwargs["device"],
            dtype=torch.int32,
        )
        self.epoch_output_idx = torch.tensor(
            output_idx, device=self.factory_kwargs["device"], dtype=torch.int32
        )
        self.epoch_output = torch.tensor(
            np.take_along_axis(self.output, output_idx - 1, axis=1),
            **self.factory_kwargs,
        )
        self.epoch_output_err = torch.tensor(
            np.take_along_axis(self.output_err, output_idx - 1, axis=1),
            **self.factory_kwargs,
        )
