import os
import json
import copy
import pathlib
import warnings
import numpy as np
import pandas as pd
from astropy.stats import mad_std
from abc import ABC, abstractmethod

from typing import List, Optional, Tuple, Union, Dict, Any
from numpy.typing import NDArray


class StellarPerceptronCore(ABC):
    """
    StellarPerceptronCore is the master class for Tensorflow and PyTorch implementation
    """

    def __init__(
        self,
        vocabs: List[str],
        backend_framewoark: str,
        vocab_tokens: List[
            int
        ] = None,  # only needed if you have a custom vocabs dictionary
        context_length: int = 30,
        embedding_dim: int = 16,
        embedding_activation: str = None,
        encoder_head_num: int = 2,
        encoder_dense_num: int = 128,
        encoder_dropout_rate: float = 0.1,
        encoder_activation: str = None,
        decoder_head_num: int = 2,
        decoder_dense_num: int = 128,
        decoder_dropout_rate: float = 0.1,
        decoder_activation: str = None,
        device: str = None,
        dtype=None,
        mixed_precision: bool = False,
        folder: str = "model_torch",
        built: bool = False,
    ) -> None:
        self._built = built
        self.backend_framewoark = backend_framewoark

        # check if vocabs is a list
        if isinstance(vocabs, np.ndarray):
            self.vocabs = vocabs.tolist()
        else:
            self.vocabs = vocabs

        # check if vocabs are unique
        if len(self.vocabs) != len(set(self.vocabs)):
            raise ValueError("Vocabs are not unique!")

        self.vocab_size = len(vocabs)
        # remember always one (the first) for special padding token
        if vocab_tokens is None:
            self.vocab_tokens = [i for i in range(1, self.vocab_size + 1)]
        else:
            self.vocab_tokens = vocab_tokens

        self._input_mean = np.zeros(np.max(self.vocab_tokens) + 1, dtype=float)
        self._input_std = np.ones(np.max(self.vocab_tokens) + 1, dtype=float)
        self._input_standardized = np.zeros(np.max(self.vocab_tokens) + 1, dtype=bool)

        self.context_length = context_length
        self.embedding_dim = embedding_dim
        self.embedding_activation = embedding_activation
        self.encoder_head_num = encoder_head_num
        self.encoder_dense_num = encoder_dense_num
        self.encoder_dropout_rate = encoder_dropout_rate
        self.encoder_activation = encoder_activation
        self.decoder_head_num = decoder_head_num
        self.decoder_dense_num = decoder_dense_num
        self.decoder_dropout_rate = decoder_dropout_rate
        self.decoder_activation = decoder_activation

        # Training parameters storage
        self.epochs = None  # how many epochs the model is supposed to train
        self.epoch = None  # how many epochs the model is trained
        self.loss = None  # last loss
        self.val_loss = None  # last validation loss
        self.learning_rate = None  # last learning rate used

        self.optimizer = None  # optimizer
        self.metrics = None  # metrics

        # Perception storage
        self._perception_memory = None
        self._last_padding_mask = None

        # ============ For PyTorch ============
        if "torch" in backend_framewoark:
            self.device = device
            self.dtype = dtype
            self.mixed_precision = mixed_precision

        self.root_folder = os.path.abspath(folder)
        # prevent any overwriting to exisitng model folder
        folder_path = pathlib.Path(self.root_folder)
        if not self._built:
            if folder_path.exists():
                raise FileExistsError(
                    f"Model folder at {self.root_folder} already existed, please rename it"
                )
            else:
                folder_path.mkdir(parents=True)

        self.system_info = {}

    @abstractmethod
    def fit(self):
        """
        For framework specific fitting of model
        """
        raise NotImplementedError

    @abstractmethod
    def _perceive_internal(self):
        """
        For framework specific perceive of information from model
        """
        raise NotImplementedError

    @abstractmethod
    def _request_internal(self):
        """
        For framework specific request of information from model
        """
        raise NotImplementedError

    @abstractmethod
    def _load_internal(self):
        """
        For framework specific loading of model
        """
        raise NotImplementedError

    @abstractmethod
    def _save_internal(self):
        """
        For framework specific saving of model
        """
        raise NotImplementedError

    def _built_only(self):
        """
        Check if the model is built, if not raise error. For functions that can only be called after model is built
        """
        if not self._built:
            raise NotImplementedError("This model is not trained")

    def clear_perception(self) -> None:
        """
        Clear the perception memory
        """
        # delete self._perception_memory in case it lives on VRAM
        del self._perception_memory
        self._perception_memory = None

    def _perception_check(self, mode: int) -> None:
        """
        Check if perception exists, if not raise error or warn depending on mode

        Parameters
        ----------
        mode : int
            1 to check if perception exists, if not raise error
            2 to check if perception exists, if exist warn
        """
        if self._perception_memory is None and mode == 1:
            raise ValueError("You did not setup a perception, so cant continue")
        elif self._perception_memory is not None and mode == 2:
            warnings.warn(
                "The existing memory in the model will be wiped and reset to your new inputs"
            )
            self.clear_perception()
        else:
            pass

    def _fit_checklist(
        self,
        inputs: NDArray,
        inputs_name: Union[NDArray, list],
        outputs_name: List[str],
        inputs_err: Optional[NDArray] = None,
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Backend framework independent fit() checklist
        """
        inputs_name = np.asarray(inputs_name)
        if inputs_name.ndim == 2 and not np.all(
            [len(set(inputs_name[:, idx])) == 1 for idx in inputs_name.shape[1]]
        ):
            raise ValueError(
                "You need to give a orderly structure inputs to me, i.e. inputs cannot be pre-emptively randomized by you but I will handle it later. You can check source code"
            )

        additional_padding_needed = self.context_length - inputs.shape[1]
        if 0 < additional_padding_needed:
            warnings.warn(
                f"Input width={inputs.shape[1]} which is smaller than context width of the model {self.context_length} so padding will be added"
            )
            inputs = self.end_zero_padding(inputs, additional_padding_needed)
            inputs_err = self.end_zero_padding(inputs_err, additional_padding_needed)
            inputs_name = self.end_zero_padding(inputs_name, additional_padding_needed)

        data_length = len(inputs)
        inputs_token = self.tokenize(inputs_name, data_length=data_length)
        standardized_inputs, inputs_token, standardized_inputs_err = self.standardize(
            inputs, inputs_token, inputs_err
        )
        outputs_tokens = self.tokenize(outputs_name)[0]
        self._built = True

        return (
            standardized_inputs,
            inputs_token,
            outputs_tokens,
            standardized_inputs_err,
        )

    def _tokenize_core_logic(self, one_str: str) -> int:
        """
        Core logic of tokenization, used by both tokenize() and tokenize_name().
        Only tokenize one string at a time
        """
        if not one_str in self.vocabs:
            raise NameError(
                f"'{one_str}' is not one of the vocabs the model know which is {self.vocabs}"
            )
        if one_str == "[pad]":
            return 0
        else:
            return self.vocab_tokens[self.vocabs.index(one_str)]

    def _set_standardization(
        self, means: NDArray, stddev: NDArray, names: List[str]
    ) -> None:
        """
        Internal use to set standardization if they are pre-computed
        """
        tokens = self.tokenize(names, data_length=1)[0]
        for idx, i in enumerate(tokens):
            self._input_mean[i] = means[idx]
            self._input_std[i] = stddev[idx]
            self._input_standardized[i] = True

    def tokenize(
        self,
        names: Union[List[Union[str, int]], NDArray[Union[np.str_, np.integer]]],
        data_length: Optional[int] = None,
    ) -> NDArray[np.integer]:
        """
        Function to tokenize names

        Parameters
        ----------
        names : Union[List[Union[str, int]], NDArray[Union[np.str_, np.integer]]]
            List of names to be tokenized
        data_length : Optional[int], optional
            If provided, the output token array will be of this length, by default None so output token array will be of same length as input names array

        Returns
        -------
        NDArray[np.integer]
            Tokenized names
        """
        names = np.atleast_2d(names)
        if len(names) > 1 and data_length is not None:
            assert (
                len(names) == data_length
            ), f"'data_length' arguement (if provided) must match 'names' array length, they are {data_length} and {len(names)} now"
        need_tiling = True
        if data_length is None:
            # need to figure out the data_length if not provided
            data_length = len(names)
        need_tiling = data_length != len(names)
        out_tokens = np.zeros(names.shape, dtype=int)  # initialize output token array

        if names.dtype == int:  # in case already tokenized, then nothing to do
            out_tokens = names
            if need_tiling:
                out_tokens = np.tile(names, (data_length, 1))
        else:  # the case where names are strings
            # in case tokens are already tiled and in nice order for every row
            # OR in case only header names are given, so do the header and tiles
            nice_order = np.all([len(set(i)) == 1 for i in names.T])
            if nice_order or need_tiling:
                _temp_names = names[0] if (nice_order and not need_tiling) else names
                out_tokens = np.tile(
                    [
                        self._tokenize_core_logic(i)
                        for i in np.atleast_1d(np.squeeze(_temp_names))
                    ],
                    (data_length, 1),
                )
            # this is the case where tokens ordering for each row is different
            else:
                # do it the slow way
                for i in np.unique(names):
                    idx = names == i
                    out_tokens[idx] = self._tokenize_core_logic(i)
        return out_tokens

    def detokenize(self, tokens: NDArray) -> NDArray:
        # need to offset by 1 index because 0 is reserved for padding
        detokenized = [self.vocabs[t - 1] for t in tokens]
        return detokenized

    def end_zero_padding(self, x: NDArray, n: int = 0) -> NDArray:
        """
        Function to padding N zeros to the end of a 2D array
        """
        if n < 0:
            raise ValueError("'n' must be non-negative")
        return np.pad(x, ((0, 0), (0, n)), mode="constant", constant_values=0)

    def padding(
        self,
        inputs: NDArray[Union[np.float32, np.float64]],
        inputs_token: NDArray[np.integer],
    ) -> Tuple[NDArray[Union[np.float32, np.float64]], NDArray[np.integer]]:
        """
        function to pad the inputs and tokens to at least the model context width
        """
        assert (
            inputs.shape == inputs_token.shape
        ), "shape of 'inputs' and 'inputs_token' arrays should be the same"
        additional_padding_needed = self.context_length - inputs.shape[1]
        if additional_padding_needed < 0:
            raise ValueError(
                f"Your inputs width={inputs.shape[1]} is larger than the context width of the model which is {self.context_length}"
            )

        padded_inputs = self.end_zero_padding(inputs, additional_padding_needed)
        padded_inputs_token = self.end_zero_padding(
            inputs_token, additional_padding_needed
        )

        return padded_inputs, padded_inputs_token

    def standardize(
        self,
        inputs: NDArray[Union[np.float32, np.float64]],
        inputs_token: NDArray[np.int_],
        inputs_error: Optional[NDArray] = None,
        dtype: np.dtype = np.float32,
    ) -> Tuple[NDArray, ...]:
        """
        Standardize input data to mean=0, stddev=1. Also set NaN to 0 and as padding

        if "inputs_error" is given, this function assume inputs_error is in the same order of inputs (i.e. inputs_token also applies to inputs_error)
        """
        # prevent changing the original array
        _inputs = copy.deepcopy(inputs)
        _inputs_token = self.tokenize(copy.deepcopy(inputs_token))
        _inputs_error = copy.deepcopy(inputs_error)

        unique_tokens = np.unique(inputs_token)

        for i in unique_tokens:
            if i != 0:  # no standardize for padding token
                if not self._input_standardized[i]:
                    # if not already standardized before, calcaulate mean and stddev again
                    self._input_mean[i] = np.nanmedian(_inputs[inputs_token == i])
                    self._input_std[i] = mad_std(
                        _inputs[inputs_token == i], ignore_nan=True
                    )
                    self._input_standardized[i] = True
                # standardize data
                _inputs[inputs_token == i] -= self._input_mean[i]
                _inputs[inputs_token == i] /= self._input_std[i]
                if _inputs_error is not None:
                    _inputs_error[inputs_token == i] /= self._input_std[i]

        # mask NaN
        nan_idx = np.isnan(inputs)
        _inputs_token = np.where(nan_idx, 0, inputs_token)
        _inputs = np.where(nan_idx, 0, _inputs)
        if _inputs_error is None:
            return _inputs, _inputs_token
        else:
            _inputs_error = np.where(nan_idx, 0, _inputs_error)
            return (
                _inputs.astype(dtype=dtype),
                _inputs_token,
                _inputs_error.astype(dtype=dtype),
            )

    def inverse_standardize(
        self,
        inputs: NDArray[Union[np.float32, np.float64]],
        inputs_token: NDArray[np.int_],
        inputs_error: Optional[NDArray[Union[np.float32, np.float64]]] = None,
    ) -> Tuple[NDArray, ...]:
        # prevent changing the original array
        _inputs = copy.deepcopy(inputs)
        _inputs_token = self.tokenize(copy.deepcopy(inputs_token))
        _inputs_error = copy.deepcopy(inputs_error)

        unique_tokens = np.unique(inputs_token)

        # if (
        #     inputs.ndim == 2 and np.squeeze(inputs_token).ndim == 1
        # ):  # assuming inputs are nicely ordered
        #     inputs_token = np.tile(inputs_token, (len(_inputs), 1))

        for i in unique_tokens:
            _inputs[inputs_token == i] *= self._input_std[i]
            _inputs[inputs_token == i] += self._input_mean[i]
            if _inputs_error is not None:
                _inputs_error[inputs_token == i] *= self._input_std[i]
        if _inputs_error is None:
            return _inputs
        else:
            return _inputs, _inputs_error

    def get_config(self) -> Dict[str, Any]:
        # get config of the network
        nn_config = {
            "backend_framewoark": self.backend_framewoark,
            "context_length": self.context_length,
            "embedding_dim": self.embedding_dim,
            "embedding_activation": self.embedding_activation,
            "encoder_head_num": self.encoder_head_num,
            "encoder_dense_num": self.encoder_dense_num,
            "encoder_dropout_rate": self.encoder_dropout_rate,
            "encoder_activation": self.encoder_activation,
            "decoder_head_num": self.decoder_head_num,
            "decoder_dense_num": self.decoder_dense_num,
            "decoder_dropout_rate": self.decoder_dropout_rate,
            "decoder_activation": self.decoder_activation,
        }
        # get config of the tokenizer
        tokenizer_config = {
            "vocabs": self.vocabs,
            "vocab_tokens": self.vocab_tokens,
        }

        # get config of normalization
        norm_config = {
            "_input_mean": self._input_mean.tolist(),
            "_input_std": self._input_std.tolist(),
            "_input_standardized": self._input_standardized.tolist(),
        }

        return {
            "nn_config": nn_config,
            "tokenizer_config": tokenizer_config,
            "norm_config": norm_config,
        }

    def save(self, folder_name: str = "model") -> None:
        """
        Backend framework independent save
        """
        pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)
        json_path = f"{folder_name}/config.json"
        if not os.path.exists(json_path):
            with open(json_path, "w") as f:
                json.dump(self.get_config(), f, indent=4)
            self._save_internal(folder_name)
        else:
            raise FileExistsError(
                "This folder seems to already has a model saved insider! Please use a different folder name."
            )

    @classmethod
    def load(cls, folder_name: str, checkpoint_epoch: int = -1, mixed_precision: bool = False, device: str = "cpu"):
        """
        Backend framework independent loading

        Parameters
        ----------
        folder_name : str
            folder name of the model
        checkpoint_epoch : int
            checkpoint epoch to load, -1 for the final model
        mixed_precision : bool
            whether to use mixed precision
        device : str
            device to load the model to
        """
        if checkpoint_epoch != -1:
            folder_name = f"{folder_name}/checkpoints/epoch_{checkpoint_epoch}"
            if not os.path.exists(folder_name):
                raise FileNotFoundError(
                    f"Checkpoint at epoch {checkpoint_epoch} not found!"
                )
        elif checkpoint_epoch < -1:
            raise ValueError("checkpoint_epoch must be >= 0")
        if not os.path.exists(folder_name):
            raise FileNotFoundError
        else:
            with open(f"{folder_name}/config.json", "r") as f:
                config = json.load(f)

        nn = cls(
            vocabs=np.array(config["tokenizer_config"]["vocabs"]),
            vocab_tokens=np.array(config["tokenizer_config"]["vocab_tokens"]),
            context_length=config["nn_config"]["context_length"],
            embedding_dim=config["nn_config"]["embedding_dim"],
            embedding_activation=config["nn_config"]["embedding_activation"],
            encoder_head_num=config["nn_config"]["encoder_head_num"],
            encoder_dense_num=config["nn_config"]["encoder_dense_num"],
            encoder_dropout_rate=config["nn_config"]["encoder_dropout_rate"],
            encoder_activation=config["nn_config"]["encoder_activation"],
            decoder_head_num=config["nn_config"]["decoder_head_num"],
            decoder_dense_num=config["nn_config"]["decoder_dense_num"],
            decoder_dropout_rate=config["nn_config"]["decoder_dropout_rate"],
            decoder_activation=config["nn_config"]["decoder_activation"],
            device=device,
            folder=folder_name,
            built=True,
        )

        # sanity check backend framework
        assert (
            nn.implemented_backend in config["nn_config"]["backend_framewoark"]
        ), f"You are loading a model trained with '{config['nn_config']['backend_framewoark']}' but you are using '{nn.implemented_backend}'"

        nn._input_mean = np.array(config["norm_config"]["_input_mean"])
        nn._input_std = np.array(config["norm_config"]["_input_std"])
        nn._input_standardized = np.array(config["norm_config"]["_input_standardized"])

        nn._load_internal(folder_name, device=device)
        nn.mixed_precision = mixed_precision

        return nn

    def perceive(
        self,
        inputs: Union[List[float], NDArray],
        inputs_token: List[Union[int, str]],
        batch_size: int = 1024,
        return_attention_scores: bool = False,
    ) -> None:
        """
        This function to initiate perception of stars in the model

        Parameters
        ----------
        inputs : array-like
            The input data to be perceived. The shape of the input data should be (n_samples, n_features).
            If it is pandas DataFrame, the column names should be vacobs understood by the model.
        inputs_token: list, optional
            Tokens or names of input data.
        batch_size: int, optional
            The batch size for neural network. Default is 1024.

        Returns
        -------
        None

        Examples
        --------
        >>> nn_model.perceive([4700, 2.5], ["teff", "logg"])
        """
        self._perception_check(mode=2)
        self._built_only()

        inputs = np.atleast_2d(inputs)
        inputs_token = np.atleast_2d(inputs_token)
        inputs_token = self.tokenize(inputs_token, data_length=len(inputs))
        inputs, inputs_token = self.standardize(inputs, inputs_token)
        inputs, inputs_token = self.padding(inputs, inputs_token)
        self._last_padding_mask = inputs_token == 0

        self._perception_memory, attention_scores = self._perceive_internal(
            inputs, inputs_token, batch_size=batch_size, return_attention_scores=return_attention_scores,
        )
        if return_attention_scores:
            return attention_scores

    def request(
        self,
        request_tokens: List[Union[int, str]],
        batch_size: int = 1024,
        return_attention_scores: bool = False,
    ):
        """
        This function to initiate perception of stars in the model
        """
        self._perception_check(mode=1)
        self._built_only()

        request_tokens = np.atleast_2d(request_tokens)

        if request_tokens.dtype.type == np.str_:
            request_tokens = self.tokenize(
                request_tokens, data_length=len(self._perception_memory)
            )

        pred, pred_err, attention_scores = self._request_internal(
            request_tokens,
            batch_size=batch_size,
            return_attention_scores=return_attention_scores,
        )
        pred, pred_err = self.inverse_standardize(pred, request_tokens, pred_err)

        all_pred = np.hstack((pred, pred_err))
        # detokenize instead of getting from arguement to make sure the name is english
        col_names = self.detokenize(request_tokens[0])
        col_names.extend([f"{i}_error" for i in col_names])
        df = pd.DataFrame(all_pred, columns=col_names)
        if return_attention_scores:
            return df, attention_scores
        else:
            return df
