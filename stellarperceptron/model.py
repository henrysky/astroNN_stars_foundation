import tqdm
import torch
import pathlib
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split

from typing import List, Optional
from numpy.typing import NDArray


from .model_core import StellarPerceptronCore
from .layers import NonLinearEmbedding, StellarPerceptronTorchModel
from .nn_utils import TrainingGenerator, robust_mean_squared_error, mean_squared_error
from .torch_utils_collect_env import get_torch_env_info


class StellarPerceptron(StellarPerceptronCore):
    """
    StellarPerceptron is a model implemented in PyTorch to demonstrate the power of Transformer-based Model.
    """

    def __init__(
        self,
        vocabs: List[str],
        vocab_tokens: List[int] = None,
        context_length: int = 30,
        embedding_dim: int = 32,
        embedding_activation=None,
        encoder_head_num: int = 2,
        encoder_dense_num: int = 128,
        encoder_dropout_rate: float = 0.1,
        encoder_activation=None,
        decoder_head_num: int = 2,
        decoder_dense_num: int = 128,
        decoder_dropout_rate: float = 0.1,
        decoder_activation=None,
        device: str = "cpu",  # PyTorch implementation only
        dtype: torch.dtype = torch.float32,  # PyTorch implementation only
        mixed_precision: bool = False,  # PyTorch implementation only
        folder: str = "model_torch",
        built: bool = False,  # do not use this arguement, it is for internal use only
    ) -> None:
        super().__init__(
            vocabs=vocabs,
            backend_framewoark=f"torch-{torch.__version__[:5]}",  # only grab version, without cpu/cuda detail
            vocab_tokens=vocab_tokens,
            context_length=context_length,
            embedding_dim=embedding_dim,
            embedding_activation=embedding_activation,
            encoder_head_num=encoder_head_num,
            encoder_dense_num=encoder_dense_num,
            encoder_dropout_rate=encoder_dropout_rate,
            encoder_activation=encoder_activation,
            decoder_head_num=decoder_head_num,
            decoder_dense_num=decoder_dense_num,
            decoder_dropout_rate=decoder_dropout_rate,
            decoder_activation=decoder_activation,
            device=device,  # PyTorch implementation only
            dtype=dtype,  # PyTorch implementation only
            mixed_precision=mixed_precision,  # PyTorch implementation only
            folder=folder,
            built=built,
        )
        self.implemented_backend = "torch"
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }  # PyTorch implementation only
        self.device_type = "cpu"
        if "cuda" in self.device:
            self.device_type = "cuda"

        self.embedding_layer = NonLinearEmbedding(
            input_dim=self.vocab_size + 1,  # plus 1 special padding token
            output_dim=self.embedding_dim,
            embeddings_initializer=torch.nn.init.xavier_uniform_,
            activation=self.embedding_activation,
            **self.factory_kwargs,
        )

        # ====================== Model initialization ======================
        self.torch_model = StellarPerceptronTorchModel(
            self.embedding_layer,
            embedding_dim=self.embedding_dim,
            encoder_head_num=self.encoder_head_num,
            encoder_dense_num=self.encoder_dense_num,
            encoder_dropout_rate=self.encoder_dropout_rate,
            encoder_activation=self.encoder_activation,
            decoder_head_num=self.decoder_head_num,
            decoder_dense_num=self.decoder_dense_num,
            decoder_dropout_rate=self.decoder_dropout_rate,
            decoder_activation=self.decoder_activation,
            **self.factory_kwargs,
        )
        self.torch_encoder = self.torch_model.torch_encoder
        self.torch_decoder = self.torch_model.torch_decoder
        # ====================== Model initialization ======================

    def _save_internal(self, folder_name: str):
        if self.optimizer is None:
            raise ValueError("Optimizer is not initialized, please (re)-train the model first")

        torch.save(
            {
                "model_state_dict": self.torch_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "optimizer": self.optimizer.__class__.__name__,
                "epoch": self.epoch,
            },
            f"{folder_name}/weights.pt",
        )

    def _load_internal(self, folder_name: str, **kwargs):
        # need to deal with gpu or not
        map_location = kwargs.get("device", "cpu")
        model_f = torch.load(f"{folder_name}/weights.pt", map_location=map_location)
        self.torch_model.load_state_dict(
            model_f["model_state_dict"],
            strict=True,
        )
        # overwrite encoder decoder from __init__ with the saved model
        self.torch_encoder = self.torch_model.torch_encoder
        self.torch_decoder = self.torch_model.torch_decoder

    def get_parameters_sum(self):
        """
        Function to count the tortal number of trainable parameters
        """
        model_parameters = filter(
            lambda p: p.requires_grad, self.torch_model.parameters()
        )
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

    def export_onnx_model(self, folder: Optional[str]=None) -> None:
        """
        Function to convert the model to ONNX format, need to split in many parts because of ONNX format limitation
        """
        torch.onnx.export(
            self.torch_encoder,
            (
                torch.randn(
                    1, self.context_length, self.embedding_dim, requires_grad=False
                ),
                torch.ones(
                    1, self.context_length, requires_grad=False, dtype=torch.bool
                ),
            ),
            "model_encoder.onnx" if folder is None else f"{folder}/model_encoder.onnx",
            export_params=True,
            input_names=["input", "mask"],
            output_names=["output"],
            dynamic_axes={  # variable length axes
                "input": {0: "batch_size"},
                "mask": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
        torch.onnx.export(
            self.torch_decoder,
            (
                torch.randn(
                    1, self.context_length, self.embedding_dim, requires_grad=False
                ),
                torch.randn(
                    1, self.context_length, self.embedding_dim, requires_grad=False
                ),
            ),
            "model_decoder.onnx" if folder is None else f"{folder}/model_decoder.onnx",
            export_params=True,
            input_names=["unit_vector", "percetion"],
            output_names=["output"],
            dynamic_axes={  # variable length axes
                "unit_vector": {0: "batch_size"},
                "percetion": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
        torch.onnx.export(
            self.embedding_layer,
            (
                torch.zeros(
                    1, self.context_length, requires_grad=False, dtype=torch.int32
                ),
                torch.randn(1, self.context_length, requires_grad=False),
            ),
            "model_embedding.onnx"
            if folder is None
            else f"{folder}/model_embedding.onnx",
            export_params=True,
            input_names=["input_tokens", "inputs"],
            output_names=["output"],
            dynamic_axes={  # variable length axes
                "input_tokens": {0: "batch_size"},
                "inputs": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
        torch.onnx.export(
            self.embedding_layer,
            (
                torch.zeros(
                    1, self.embedding_dim, requires_grad=False, dtype=torch.int32
                )
            ),
            "model_unitvec.onnx" if folder is None else f"{folder}/model_unitvec.onnx",
            export_params=True,
            input_names=["input_tokens"],
            output_names=["output"],
            dynamic_axes={  # variable length axes
                "input_tokens": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )

    def fit(
        self,
        inputs: NDArray,
        inputs_name: NDArray,
        outputs_name: List[str],
        inputs_err: Optional[NDArray]=None,
        # max number of tokens will be turned to padding during training
        outputs_padding: int = 11,
        batch_size: int = 64,
        # batch size to use for validation compared to training batch size
        val_batchsize_factor: int = 5,
        epochs: int = 32,
        validation_split: float = 0.1,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau,
        checkpoint_every_n_epochs: int = 0,  # save checkpoint every n epochs, put 0 to disable
        terminate_on_nan: bool = True
    ) -> None:
        # always scale the gradients if using cuda
        gradient_scaler = torch.cuda.amp.GradScaler(enabled=self.device_type == "cuda")
        self.epochs = epochs
        if inputs_err is None:
            inputs_err = np.zeros_like(inputs)

        # check checkpoint_every_n_epochs
        if checkpoint_every_n_epochs < 0:
            raise ValueError("checkpoint_every_n_epochs can not be less than zero")
        else:
            pathlib.Path(f"{self.root_folder}/checkpoints").mkdir(
                parents=True, exist_ok=True
            )

        training_log_f = open(f"{self.root_folder}/training.log", "w")
        training_log_f.write(f"Batch Size: {batch_size}\n")
        training_log_f.write("====================================\n")


        training_csv_metrics_f = open(f"{self.root_folder}/training_metrics.csv", "w")
        training_csv_metrics_f.write("time,loss,mse_loss,val_loss,val_mse_loss,lr\n")

        system_info_f = open(f"{self.root_folder}/training_system_info.log", "w")
        system_info_f.write(get_torch_env_info())
        system_info_f.close()

        (
            standardized_inputs,
            inputs_token,
            outputs_tokens,
            standardized_inputs_err,
        ) = self._fit_checklist(
            inputs=inputs,
            inputs_name=inputs_name,
            outputs_name=outputs_name,
            inputs_err=inputs_err,
        )

        (
            X_train,
            X_val,
            X_train_token,
            X_val_token,
            X_Err_train,
            X_Err_val,
        ) = train_test_split(
            standardized_inputs,
            inputs_token,
            standardized_inputs_err,
            test_size=validation_split,
        )

        training_generator = TrainingGenerator(
            batch_size=batch_size,
            data={
                "input": X_train,
                "input_token": X_train_token,
                "output": X_train,
                "output_err": X_Err_train,
            },
            possible_output_tokens=outputs_tokens,
            outputs_padding=outputs_padding,
            input_length=self.context_length,
            factory_kwargs=self.factory_kwargs,
        )

        val_generator = TrainingGenerator(
            batch_size=batch_size * val_batchsize_factor,
            data={
                "input": X_val,
                "input_token": X_val_token,
                "output": X_val,
                "output_err": X_Err_val,
            },
            possible_output_tokens=outputs_tokens,
            outputs_padding=outputs_padding,
            input_length=self.context_length,
            shuffle=False,  # no need to shuffle for validation
            factory_kwargs=self.factory_kwargs,
        )

        scheduler = lr_scheduler(self.optimizer)

        # ====================== Training logic ======================
        elapsed_time = 0
        with tqdm.tqdm(range(epochs), unit="epoch") as pbar:
            for epoch in pbar:
                self.epoch = epoch + 1
                # print(f"Epoch {self.epoch}/{self.epochs}")
                training_log_f.write(f"Epoch {self.epoch}/{self.epochs}\n")

                self.torch_model.train()
                running_loss = 0.0
                running_mse_loss = 0.0
                last_loss = 0.0

                # order: input, input_token, label_token, label, label_err
                for batch_num, (
                    inputs,
                    input_token,
                    label_token,
                    label,
                    label_err,
                ) in enumerate(training_generator):
                    # reset gradient for every batch
                    self.optimizer.zero_grad()
                    with torch.autocast(
                        device_type=self.device_type,
                        enabled=self.mixed_precision,
                    ):
                        outputs, outputs_logvar = self.torch_model(
                            inputs,
                            input_token,
                            label_token,
                        )

                        loss = robust_mean_squared_error(
                            label,
                            outputs[:, :, 0],
                            outputs_logvar[:, :, 0],
                            labels_err=label_err,
                        )
                        loss_mse = mean_squared_error(
                            outputs[:, :, 0], 
                            label
                            )
                    gradient_scaler.scale(loss).backward()
                    gradient_scaler.step(self.optimizer)
                    gradient_scaler.update()
                    running_loss += loss.item()
                    running_mse_loss += loss_mse.item()

                last_loss = running_loss / (batch_num + 1)
                last_mse_loss = running_mse_loss / (batch_num + 1)
                training_generator.on_epoch_end()
                # ======== epoch level validation ========
                self.torch_model.eval()
                running_vloss = 0.0
                running_vloss_mse = 0.0
                with torch.inference_mode():
                    # order: input, input_token, label_token, label, label_err
                    for batch_num, (
                        inputs,
                        input_token,
                        label_token,
                        label,
                        label_err,
                    ) in enumerate(val_generator):
                        voutputs, voutputs_logvar = self.torch_model(
                            inputs,
                            input_token,
                            label_token,
                        )
                        vloss = robust_mean_squared_error(
                            label,
                            voutputs[:, :, 0],
                            voutputs_logvar[:, :, 0],
                            labels_err=label_err,
                        )
                        vloss_mse = mean_squared_error(
                            voutputs[:, :, 0],
                            label,
                        )
                        running_vloss += vloss.item()
                        running_vloss_mse += vloss_mse.item()

                avg_vloss = running_vloss / (batch_num + 1)
                avg_vloss_mse = running_vloss_mse / (batch_num + 1)

                # store loss, val_loss and learning rate
                self.loss = last_loss
                self.val_loss = avg_vloss
                self.learning_rate = self.optimizer.param_groups[-1]["lr"]
                val_generator.on_epoch_end()

                # ======== post-epoch activity ========
                scheduler.step()
                lr_fmt = np.format_float_scientific(
                    self.learning_rate, precision=4, unique=False
                )
                temp_time = pbar.format_dict["elapsed"] - elapsed_time
                elapsed_time = pbar.format_dict["elapsed"]
                training_log_f.write(
                    f"elapsed: {str(timedelta(seconds=elapsed_time))}s - rate: {temp_time:.2f}s - loss: {last_loss:.4f} - mse_loss: {last_mse_loss:.4f} val_loss {avg_vloss:.4f} - val_mse_loss {avg_vloss_mse:.4f} - lr: {lr_fmt}\n"
                )
                training_log_f.flush()
                training_csv_metrics_f.write(
                    f"{temp_time},{last_loss},{last_mse_loss},{avg_vloss},{avg_vloss_mse},{lr_fmt}\n"
                )
                training_csv_metrics_f.flush()

                if terminate_on_nan and np.isnan(last_loss):
                    raise ValueError("Loss is NaN, hence training terminated!")

                if checkpoint_every_n_epochs > 0:
                    # always save the one right after first epoch
                    if self.epoch % checkpoint_every_n_epochs == 0 or self.epoch == 1:
                        folder_path = (
                            f"{self.root_folder}/checkpoints/epoch_{self.epoch}"
                        )
                        self.save(folder_name=folder_path)
            # ====================== Training logic ======================

        training_log_f.close()
        training_csv_metrics_f.close()

    def _perceive_internal(self, inputs, inputs_token, batch_size, return_attention_scores=False, inference_mode=True):
        self.torch_model.eval()
        with torch.inference_mode(mode=inference_mode):
            inputs_token = torch.as_tensor(
                inputs_token, device=self.factory_kwargs["device"], dtype=torch.int32
            )
            input_embedded = self.embedding_layer(
                inputs_token,
                torch.atleast_3d(torch.as_tensor(inputs, **self.factory_kwargs)),
            )
            padding_mask = torch.eq(inputs_token, torch.zeros_like(inputs_token))
            self._last_padding_mask = padding_mask
            data_length = len(inputs)
            if return_attention_scores:
                attention_scores = np.zeros(
                    (data_length, self.context_length, self.context_length)
                )
            else:
                # in case you dont want and you are doing a large amount of stars
                attention_scores = None
            num_batch = data_length // batch_size
            num_batch_remainder = data_length % batch_size
            if num_batch == 0:  # if smaller than batch_size, then do all at once
                perception = self.torch_encoder(input_embedded, mask=padding_mask)
                if return_attention_scores:
                    _last_padding_mask = self._last_padding_mask.detach().to("cpu").numpy()
                    last_attention_scores = self.torch_encoder.last_attention_scores.detach().to("cpu").numpy()
                    attention_scores = np.where(np.tile(np.atleast_3d(_last_padding_mask), (1, 1, self.context_length)), 0, last_attention_scores[:, : self.context_length, : self.context_length])
            else:
                # TODO: need to handle attention score in this case
                if return_attention_scores:
                    raise NotImplementedError(
                        "return_attention_scores not implemented for batched inputs yet"
                    )
                with torch.autocast(
                    device_type=self.device_type,
                    enabled=self.mixed_precision,
                ):
                    perception = [
                        self.torch_encoder(
                            input_embedded[
                                i * batch_size : i * batch_size + batch_size
                            ],
                            mask=padding_mask[
                                i * batch_size : i * batch_size + batch_size
                            ],
                        )
                        for i in range(num_batch)
                    ]
                if num_batch_remainder > 0:
                    # do the remainder
                    perception.extend(
                        [
                            self.torch_encoder(
                                input_embedded[num_batch * batch_size :],
                                mask=padding_mask[num_batch * batch_size :],
                            )
                        ]
                    )
                perception = torch.concat(perception)

            if return_attention_scores:
                attention_scores /= np.sum(attention_scores, axis=1, keepdims=True)
            return perception, attention_scores
        
    def _request_internal(
        self, request_tokens, batch_size, return_attention_scores=False
    ):
        self.torch_model.eval()
        with torch.inference_mode():
            data_length = len(self._perception_memory)
            pred = np.zeros((data_length, request_tokens.shape[1]))
            pred_err = np.zeros((data_length, request_tokens.shape[1]))
            perception = torch.as_tensor(self._perception_memory, **self.factory_kwargs)
            if return_attention_scores:
                attention_scores = np.zeros(
                    (data_length, self.context_length, request_tokens.shape[1])
                )
            else:
                # in case you dont want and you are doing a large amount of stars
                attention_scores = None
            # now we only need to loop decoder
            request_tokens_num = request_tokens.shape[1]
            request_tokens = torch.as_tensor(
                request_tokens, device=self.factory_kwargs["device"], dtype=torch.int32
            )
            data_len = len(self._perception_memory)
            num_batch = data_len // batch_size
            num_batch_remainder = data_len % batch_size
            unit_vectors = torch.empty(
                (data_len, request_tokens_num, self.embedding_dim),
                **self.factory_kwargs,
            )
            for idx in range(request_tokens_num):
                unit_vectors[:, idx : idx + 1] = self.embedding_layer(
                    request_tokens[:, idx : idx + 1]
                )

            if num_batch == 0:  # if smaller than batch_size, then do all at once
                for idx in range(request_tokens_num):
                    _pred, _pred_logvar = self.torch_decoder(
                        torch.as_tensor(
                            unit_vectors[:, idx : idx + 1], **self.factory_kwargs
                        ),
                        perception,
                        self._last_padding_mask,
                    )
                    pred[:, idx] = np.squeeze(_pred.detach().to("cpu").numpy())
                    pred_err[:, idx] = np.sqrt(
                        np.exp(np.squeeze(_pred_logvar.detach().to("cpu").numpy()))
                    )
                    if return_attention_scores:
                        _last_padding_mask = self._last_padding_mask.detach().to("cpu").numpy()
                        last_attention_scores = self.torch_decoder.last_attention_scores.detach().to("cpu").numpy()
                        attention_scores[:, :, idx] = np.where(
                            _last_padding_mask,
                            0,
                            last_attention_scores[:, :, : self.context_length],
                        )
            else:
                for idx in range(request_tokens_num):
                    for i in range(num_batch):
                        with torch.autocast(
                            device_type=self.device_type,
                            enabled=self.mixed_precision,
                        ):
                            _pred, _pred_logvar = self.torch_decoder(
                                torch.as_tensor(
                                    unit_vectors[
                                        i * batch_size : i * batch_size + batch_size,
                                        idx : idx + 1,
                                    ],
                                    **self.factory_kwargs,
                                ),
                                perception[
                                    i * batch_size : i * batch_size + batch_size
                                ],
                                self._last_padding_mask[
                                    i * batch_size : i * batch_size + batch_size
                                ],
                            )
                        pred[
                            i * batch_size : i * batch_size + batch_size, idx
                        ] = np.squeeze(_pred.detach().to("cpu").numpy())
                        pred_err[
                            i * batch_size : i * batch_size + batch_size, idx
                        ] = np.sqrt(
                            np.exp(np.squeeze(_pred_logvar.detach().to("cpu").numpy()))
                        )
                        if return_attention_scores:
                            _last_padding_mask = self._last_padding_mask.detach().to("cpu").numpy()
                            last_attention_scores = self.torch_decoder.last_attention_scores.detach().to("cpu").numpy()
                            attention_scores[i * batch_size : i * batch_size + batch_size, :, idx] = np.where(_last_padding_mask, 0, last_attention_scores[:, :, : self.context_length],)
                if num_batch_remainder > 0:
                    # do the remainder
                    for idx in range(request_tokens_num):
                        _pred, _pred_logvar = self.torch_decoder(
                            torch.as_tensor(
                                unit_vectors[num_batch * batch_size :, idx : idx + 1],
                                **self.factory_kwargs,
                            ),
                            perception[num_batch * batch_size :],
                            self._last_padding_mask[num_batch * batch_size :],
                        )
                        pred[num_batch * batch_size :, idx] = np.squeeze(
                            _pred.detach().to("cpu").numpy()
                        )
                        pred_err[num_batch * batch_size :, idx] = np.sqrt(
                            np.exp(np.squeeze(_pred_logvar.detach().to("cpu").numpy()))
                        )
                        if return_attention_scores:
                            _last_padding_mask = self._last_padding_mask.detach().to("cpu").numpy()
                            last_attention_scores = self.torch_decoder.last_attention_scores.detach().to("cpu").numpy()
                            attention_scores[
                                num_batch * batch_size :, :, idx
                            ] = np.where(
                                _last_padding_mask,
                                0,
                                last_attention_scores[:, :, : self.context_length],
                            )
            if return_attention_scores:
                attention_scores /= np.sum(attention_scores, axis=1, keepdims=True)
            return pred, pred_err, attention_scores
