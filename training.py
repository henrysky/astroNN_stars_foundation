import h5py
import torch
import numpy as np

from stellarperceptron.model import StellarPerceptron


# ================== hardware-related settings ==================
device = "cuda:0"  # "cpu" for CPU or "cuda:x" for a NVIDIA GPU
mixed_precision = True  # use mixed precision training for CUDA
torch.backends.cuda.matmul.allow_tf32 = True  # use tf32 for CUDA matrix multiplication
torch.backends.cudnn.allow_tf32 = True  # use tf32 for CUDNN
# ================== hardware-related settings ==================

# ================== model-related settings ==================
context_length = 64  # model context window length
embedding_dim = 128  # embedding dimension
# ================== model-related settings ==================

# ================== training-related settings ==================
learning_rate = 1e-4  # initial learning rate
learning_rate_min = 1e-10  # minimum learning rate allowed
batch_size = 1024  # batch size
epochs = 4096  # number of epochs to train
cosine_annealing_t0 = 512  # cosine annealing restart length in epochs
checkpoint_every_n_epochs = 128  # save a checkpoint every n epochs
save_model_to_folder = "./model_torch/"  # folder to save the model
# ================== training-related settings ==================

# load training data
xp_apogee = h5py.File("./data_files/training_set.h5", mode="r")
xp_relevancy = xp_apogee["raw"]["xp_relevancy"][()]
xp_coeffs_gnorm = xp_apogee["raw"]["xp_coeffs_gnorm"][()]
xp_coeffs_err_gnorm = xp_apogee["raw"]["xp_coeffs_gnorm_err"][()]

# propagate to deal with 53, 54, 108, 109 NaN issues
xp_relevancy[:, 52:55] = xp_relevancy[:, 51:52]
xp_relevancy[:, 107:110] = xp_relevancy[:, 106:107]
xp_coeffs_gnorm[~xp_relevancy] = np.nan
xp_coeffs_err_gnorm[~xp_relevancy] = np.nan

training_labels = np.column_stack(
    [
        xp_coeffs_gnorm,
        xp_apogee["raw"]["bprp"][()],
        xp_apogee["raw"]["jh"][()],
        xp_apogee["raw"]["jk"][()],
        xp_apogee["raw"]["teff"][()],
        xp_apogee["raw"]["logg"][()],
        xp_apogee["raw"]["m_h"][()],
        xp_apogee["raw"]["logc19"][()],
        xp_apogee["raw"]["g_fakemag"][()],
    ]
)

print("Number of training stars: ", len(training_labels))

training_labels_err = np.column_stack(
    [
        xp_coeffs_err_gnorm,
        xp_apogee["raw"]["bprp_err"][()],
        xp_apogee["raw"]["jh_err"][()],
        xp_apogee["raw"]["jk_err"][()],
        xp_apogee["raw"]["teff_err"][()],
        xp_apogee["raw"]["logg_err"][()],
        xp_apogee["raw"]["m_h_err"][()],
        xp_apogee["raw"]["logc19_err"][()],
        xp_apogee["raw"]["g_fakemag_err"][()],
    ]
)
xp_apogee.close()
training_labels_err = np.where(np.isnan(training_labels_err), 0.0, training_labels_err)

obs_names = [
    *[f"bp{i}" for i in range(55)],
    *[f"rp{i}" for i in range(55)],
    "bprp",
    "jh",
    "jk",
    "teff",
    "logg",
    "m_h",
    "logebv",
    "g_fakemag",
]

nn_model = StellarPerceptron(
    vocabs=obs_names,
    context_length=context_length,
    embedding_dim=embedding_dim,
    embedding_activation="gelu",
    encoder_head_num=16,
    encoder_dense_num=1024,
    encoder_dropout_rate=0.1,
    encoder_activation="gelu",
    decoder_head_num=16,
    decoder_dense_num=3096,
    decoder_dropout_rate=0.1,
    decoder_activation="gelu",
    device=device,
    mixed_precision=mixed_precision,
    folder=save_model_to_folder,
)

nn_model.optimizer = torch.optim.AdamW(
    nn_model.torch_model.parameters(), lr=learning_rate
)

# There is no output labels, this model only has one output node depending what information you request
# Here we choose a set of labels from inputs as possible information request to quickly train this model
# In principle, any labels in inputs can be requested in output
nn_model.fit(
    # give all because some of them will be randomly chosen shuffled in random order for each stars in each epoch
    inputs=training_labels,
    inputs_name=obs_names,
    inputs_err=training_labels_err,
    # during training, one of these will be randomly chosen for each stars in each epoch
    outputs_name=[
        *[f"bp{i}" for i in range(55)],
        *[f"rp{i}" for i in range(55)],
        "jh",
        "teff",
        "logg",
        "m_h",
        "logebv",
        "g_fakemag",
    ],
    outputs_padding=60,
    batch_size=batch_size,
    val_batchsize_factor=10,
    epochs=epochs,
    lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cosine_annealing_t0,
        T_mult=1,
        eta_min=learning_rate_min,
        last_epoch=-1,  # means really the last epoch ever
    ),
    terminate_on_nan=True,
    checkpoint_every_n_epochs=checkpoint_every_n_epochs,
)

# ==================== save final trained model ====================
nn_model.save(save_model_to_folder)
