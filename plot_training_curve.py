import matplotlib.pyplot as plt

# =========================================================
# Fine-tune: lr=1e-4, bs=16, beam=1
# =========================================================
epochs_lr1e4_bs16 = list(range(1, 11))
train_loss_lr1e4_bs16 = [
    2.985339, 0.566347, 0.305088, 0.220332, 0.180071,
    0.157661, 0.142896, 0.135687, 0.131838, 0.130545
]
dev_loss_lr1e4_bs16 = [
    0.609692, 0.254472, 0.162505, 0.129319, 0.110085,
    0.098698, 0.092137, 0.088682, 0.087471, 0.087222
]
rec_f1_lr1e4_bs16 = [
    0.118098, 0.234802, 0.344340, 0.402315, 0.447067,
    0.484107, 0.492759, 0.498269, 0.515127, 0.513807
]
rec_em_lr1e4_bs16 = [
    0.118026, 0.214592, 0.285408, 0.334764, 0.366953,
    0.390558, 0.399142, 0.409871, 0.420601, 0.420601
]
sql_err_lr1e4_bs16 = [
    97.21, 61.59, 36.70, 35.41, 31.12,
    30.26, 30.04, 31.12, 29.40, 29.40
]


# =========================================================
# Fine-tune: lr=1e-3, bs=32, beam=2
# =========================================================
epochs_lr1e3_bs32 = list(range(1, 11))
train_loss_lr1e3_bs32 = [
    2.134821, 0.233329, 0.106545, 0.070535, 0.051612,
    0.041568, 0.035328, 0.032882, 0.029775, 0.028861
]
dev_loss_lr1e3_bs32 = [
    0.284377, 0.097667, 0.061286, 0.042837, 0.035391,
    0.030061, 0.027131, 0.025912, 0.024894, 0.024916
]
rec_f1_lr1e3_bs32 = [
    0.279520, 0.474396, 0.589741, 0.685233, 0.733104,
    0.765740, 0.770944, 0.772750, 0.790700, 0.787608
]
rec_em_lr1e3_bs32 = [
    0.242489, 0.364807, 0.493562, 0.618026, 0.673820,
    0.725322, 0.731760, 0.742489, 0.761803, 0.759657
]
sql_err_lr1e3_bs32 = [
    55.58, 23.82, 18.24, 16.31, 16.52,
    13.30, 14.59, 11.80, 12.02, 11.80
]


# =========================================================
# Encoder-only (freeze decoder): lr=1e-3, bs=32, beam=2
# =========================================================
epochs_encoder_only = list(range(1, 11))
train_loss_encoder_only = [
    4.403755, 2.172352, 1.417763, 1.134430,
    0.980572, 0.895978, 0.836759, 0.802548,
    0.802548, 0.802548
]
dev_loss_encoder_only = [
    2.316625, 1.141728, 0.794251, 0.632346,
    0.537061, 0.502017, 0.460455, 0.445764,
    0.445764, 0.445764
]
rec_f1_encoder_only = [0.118026] * 10
rec_em_encoder_only = [0.118026] * 10
sql_err_encoder_only = [
    100.00, 100.00, 100.00, 100.00,
    99.14, 99.36, 95.92, 96.35,
    96.35, 96.35
]


# =========================================================
# Decoder-only (freeze encoder): lr=1e-3, bs=32, beam=2
# =========================================================
epochs_decoder_only = list(range(1, 11))
train_loss_decoder_only = [
    2.214109, 0.264178, 0.124389, 0.084731, 0.067935,
    0.055110, 0.047895, 0.043690, 0.041513, 0.039998
]
dev_loss_decoder_only = [
    0.316364, 0.109129, 0.069253, 0.055191, 0.043178,
    0.039549, 0.036033, 0.032687, 0.032419, 0.032249
]
rec_f1_decoder_only = [
    0.263891, 0.410820, 0.507750, 0.595188, 0.662349,
    0.687986, 0.703927, 0.723636, 0.738868, 0.742030
]
rec_em_decoder_only = [
    0.223176, 0.300429, 0.422747, 0.502146, 0.560086,
    0.607296, 0.620172, 0.663090, 0.675966, 0.678112
]
sql_err_decoder_only = [
    54.94, 22.10, 29.40, 19.74, 10.30,
    13.95, 10.94, 11.16, 10.09, 9.66
]


# =========================================================
# Train only decoder layers 0,1,2 (freeze encoder + decoder 3-5 + embeddings/head)
# lr=1e-3, bs=32, beam=2
# =========================================================
epochs_dec012 = list(range(1, 11))
train_loss_dec012 = [
    2.899079, 0.474605, 0.215403, 0.149686, 0.117926,
    0.100664, 0.090405, 0.083253, 0.079904, 0.078851
]
dev_loss_dec012 = [
    0.571734, 0.191160, 0.117004, 0.089153, 0.075975,
    0.066206, 0.060401, 0.058203, 0.057164, 0.056996
]
rec_f1_dec012 = [
    0.118026, 0.287981, 0.393400, 0.434861, 0.510400,
    0.510923, 0.541426, 0.537072, 0.559222, 0.561140
]
rec_em_dec012 = [
    0.118026, 0.227468, 0.326180, 0.345494, 0.416309,
    0.414163, 0.452790, 0.454936, 0.469957, 0.472103
]
sql_err_dec012 = [
    98.28, 35.19, 27.90, 22.10, 13.09,
    16.74, 11.16, 13.52, 13.30, 13.52
]


# =========================================================
# Plot helper
# =========================================================
def plot_metric(title, ylabel, series):
    plt.figure(figsize=(8, 5))
    for x, y, label in series:
        plt.plot(x, y, marker="o", linewidth=2, label=label)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim(1, 10)  # Ensure epochs 1-10 are visible
    plt.xticks(range(1, 11))  # Show all epoch numbers 1-10
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =========================================================
# Plots
# =========================================================
plot_metric(
    "Training Loss vs Epoch",
    "Training Loss",
    [
        (epochs_lr1e4_bs16, train_loss_lr1e4_bs16, "lr=1e-4, bs=16, beam=1"),
        (epochs_lr1e3_bs32, train_loss_lr1e3_bs32, "lr=1e-3, bs=32, beam=2"),
        (epochs_encoder_only, train_loss_encoder_only, "encoder-only (lr=1e-3, bs=32, beam=2)"),
        (epochs_decoder_only, train_loss_decoder_only, "decoder-only (lr=1e-3, bs=32, beam=2)"),
        (epochs_dec012, train_loss_dec012, "decoder first 3 layers (lr=1e-3, bs=32, beam=2)"),
    ],
)

plot_metric(
    "Validation Loss vs Epoch",
    "Validation Loss",
    [
        (epochs_lr1e4_bs16, dev_loss_lr1e4_bs16, "lr=1e-4, bs=16, beam=1"),
        (epochs_lr1e3_bs32, dev_loss_lr1e3_bs32, "lr=1e-3, bs=32, beam=2"),
        (epochs_encoder_only, dev_loss_encoder_only, "encoder-only (lr=1e-3, bs=32, beam=2)"),
        (epochs_decoder_only, dev_loss_decoder_only, "decoder-only (lr=1e-3, bs=32, beam=2)"),
        (epochs_dec012, dev_loss_dec012, "decoder first 3 layers (lr=1e-3, bs=32, beam=2)"),
    ],
)

plot_metric(
    "Record F1 vs Epoch",
    "Record F1",
    [
        (epochs_lr1e4_bs16, rec_f1_lr1e4_bs16, "lr=1e-4, bs=16, beam=1"),
        (epochs_lr1e3_bs32, rec_f1_lr1e3_bs32, "lr=1e-3, bs=32, beam=2"),
        (epochs_encoder_only, rec_f1_encoder_only, "encoder-only (lr=1e-3, bs=32, beam=2)"),
        (epochs_decoder_only, rec_f1_decoder_only, "decoder-only (lr=1e-3, bs=32, beam=2)"),
        (epochs_dec012, rec_f1_dec012, "decoder first 3 layers (lr=1e-3, bs=32, beam=2)"),
    ],
)

plot_metric(
    "Record EM (Exact Match) vs Epoch",
    "Record EM (Exact Match)",
    [
        (epochs_lr1e4_bs16, rec_em_lr1e4_bs16, "lr=1e-4, bs=16, beam=1"),
        (epochs_lr1e3_bs32, rec_em_lr1e3_bs32, "lr=1e-3, bs=32, beam=2"),
        (epochs_encoder_only, rec_em_encoder_only, "encoder-only (lr=1e-3, bs=32, beam=2)"),
        (epochs_decoder_only, rec_em_decoder_only, "decoder-only (lr=1e-3, bs=32, beam=2)"),
        (epochs_dec012, rec_em_dec012, "decoder first 3 layers (lr=1e-3, bs=32, beam=2)"),
    ],
)

plot_metric(
    "SQL Error Rate vs Epoch",
    "SQL Error (%)",
    [
        (epochs_lr1e4_bs16, sql_err_lr1e4_bs16, "lr=1e-4, bs=16, beam=1"),
        (epochs_lr1e3_bs32, sql_err_lr1e3_bs32, "lr=1e-3, bs=32, beam=2"),
        (epochs_encoder_only, sql_err_encoder_only, "encoder-only (lr=1e-3, bs=32, beam=2)"),
        (epochs_decoder_only, sql_err_decoder_only, "decoder-only (lr=1e-3, bs=32, beam=2)"),
        (epochs_dec012, sql_err_dec012, "decoder first 3 layers (lr=1e-3, bs=32, beam=2)"),
    ],
)



