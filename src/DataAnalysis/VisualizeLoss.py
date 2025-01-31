import matplotlib.pyplot as plt

loss_data = [
    # Training 1 (Warm-up):
    #{"policy_head_loss": 7.0405, "value_head_loss": 0.1040, "loss": 7.1446},
    #{"policy_head_loss": 4.7344, "value_head_loss": 0.1010, "loss": 4.8354},
    #{"policy_head_loss": 4.5000, "value_head_loss": 0.1006, "loss": 4.6006},
    #{"policy_head_loss": 4.6487, "value_head_loss": 0.1006, "loss": 4.7493},
    #{"policy_head_loss": 4.9121, "value_head_loss": 0.1007, "loss": 5.0128},
    # Training 2 (Warm-up):
    #{"policy_head_loss": 4.9230, "value_head_loss": 0.1980, "loss": 5.1210},
    #{"policy_head_loss": 4.1115, "value_head_loss": 0.1976, "loss": 4.3091},
    #{"policy_head_loss": 3.7440, "value_head_loss": 0.1975, "loss": 3.9415},
    #{"policy_head_loss": 3.7112, "value_head_loss": 0.1975, "loss": 3.9088},
    #{"policy_head_loss": 3.3303, "value_head_loss": 0.1975, "loss": 3.5278},
    # Training 3 (Warm-up):
    #{"policy_head_loss": 4.1875, "value_head_loss": 0.1437, "loss": 4.3312},
    #{"policy_head_loss": 3.5763, "value_head_loss": 0.1437, "loss": 3.7200},
    #{"policy_head_loss": 3.2409, "value_head_loss": 0.1437, "loss": 3.3846},
    #{"policy_head_loss": 2.9771, "value_head_loss": 0.1437, "loss": 3.1208},
    #{"policy_head_loss": 3.1077, "value_head_loss": 0.1436, "loss": 3.2514},
    # Training 4 (Warm-up):
    #{"policy_head_loss": 4.0512, "value_head_loss": 0.1818, "loss": 4.2330},
    #{"policy_head_loss": 3.4012, "value_head_loss": 0.1817, "loss": 3.5829},
    #{"policy_head_loss": 2.9978, "value_head_loss": 0.1817, "loss": 3.1794},
    #{"policy_head_loss": 2.6926, "value_head_loss": 0.1817, "loss": 2.8743},
    #{"policy_head_loss": 2.4299, "value_head_loss": 0.1817, "loss": 2.6116},
    # Training 5 (Warm-up):
    #{"policy_head_loss": 3.4508, "value_head_loss": 0.1742, "loss": 3.6251},
    #{"policy_head_loss": 2.9447, "value_head_loss": 0.1742, "loss": 3.1188},
    #{"policy_head_loss": 2.5990, "value_head_loss": 0.1742, "loss": 2.7732},
    #{"policy_head_loss": 2.2968, "value_head_loss": 0.1742, "loss": 2.4710},
    #{"policy_head_loss": 2.0501, "value_head_loss": 0.1742, "loss": 2.2242},

    # Training 1 (Main):
    {"policy_head_loss": 67.8038, "value_head_loss": 0.2016, "loss": 68.0053},
    {"policy_head_loss": 60.0083, "value_head_loss": 0.2016, "loss": 60.2098},
    {"policy_head_loss": 317.7239, "value_head_loss": 0.2017, "loss": 317.9258},
    {"policy_head_loss": 880.9358, "value_head_loss": 0.2015, "loss": 881.1365},
    {"policy_head_loss": 739.5477, "value_head_loss": 0.2017, "loss": 739.7493},
    {"policy_head_loss": 724.9418, "value_head_loss": 0.2016, "loss": 725.1432},
    {"policy_head_loss": 676.8655, "value_head_loss": 0.2016, "loss": 677.0674},
    {"policy_head_loss": 690.6332, "value_head_loss": 0.2016, "loss": 690.8348},
    {"policy_head_loss": 676.1977, "value_head_loss": 0.2018, "loss": 676.3999},
    {"policy_head_loss": 671.0361, "value_head_loss": 0.2012, "loss": 671.2375},
    # Training 2 (Main):
    {"policy_head_loss": 601.5579, "value_head_loss": 0.1950, "loss": 601.7527},
    {"policy_head_loss": 624.1627, "value_head_loss": 0.1949, "loss": 624.3575},
    {"policy_head_loss": 645.6597, "value_head_loss": 0.1951, "loss": 645.8546},
    {"policy_head_loss": 614.9797, "value_head_loss": 0.1950, "loss": 615.1744},
    {"policy_head_loss": 620.1898, "value_head_loss": 0.1948, "loss": 620.3846},
    {"policy_head_loss": 580.6428, "value_head_loss": 0.1950, "loss": 580.8375},
    {"policy_head_loss": 585.0971, "value_head_loss": 0.1948, "loss": 585.2917},
    {"policy_head_loss": 550.0498, "value_head_loss": 0.1948, "loss": 550.2445},
    {"policy_head_loss": 522.9457, "value_head_loss": 0.1947, "loss": 523.1408},
    {"policy_head_loss": 519.9294, "value_head_loss": 0.1947, "loss": 520.1243},
    # Training 3 (Main):
    {"policy_head_loss": 577.4250, "value_head_loss": 0.2014, "loss": 577.6266},
    {"policy_head_loss": 544.3767, "value_head_loss": 0.2013, "loss": 544.5783},
    {"policy_head_loss": 556.9100, "value_head_loss": 0.2016, "loss": 557.1113},
    {"policy_head_loss": 532.4692, "value_head_loss": 0.2012, "loss": 532.6705},
    {"policy_head_loss": 528.6878, "value_head_loss": 0.2013, "loss": 528.8889},
    {"policy_head_loss": 515.1212, "value_head_loss": 0.2013, "loss": 515.3225},
    {"policy_head_loss": 488.4434, "value_head_loss": 0.2012, "loss": 488.6449},
    {"policy_head_loss": 477.4789, "value_head_loss": 0.2013, "loss": 477.6803},
    {"policy_head_loss": 458.6414, "value_head_loss": 0.2012, "loss": 458.8423},
    {"policy_head_loss": 444.7288, "value_head_loss": 0.2015, "loss": 444.9302},
    # Training 4 (Main):
    {"policy_head_loss": 485.2839, "value_head_loss": 0.1964, "loss": 485.4801},
    {"policy_head_loss": 466.7669, "value_head_loss": 0.1963, "loss": 466.9638},
    {"policy_head_loss": 454.3549, "value_head_loss": 0.1964, "loss": 454.5514},
    {"policy_head_loss": 443.4609, "value_head_loss": 0.1964, "loss": 443.6574},
    {"policy_head_loss": 440.7303, "value_head_loss": 0.1965, "loss": 440.9266},
    {"policy_head_loss": 417.5954, "value_head_loss": 0.1963, "loss": 417.7919},
    {"policy_head_loss": 408.3515, "value_head_loss": 0.1961, "loss": 408.5476},
    {"policy_head_loss": 392.0862, "value_head_loss": 0.1963, "loss": 392.2828},
    {"policy_head_loss": 369.2449, "value_head_loss": 0.1963, "loss": 369.4414},
    {"policy_head_loss": 357.6480, "value_head_loss": 0.1963, "loss": 357.8441},
    # Training 5 (Main):
    {"policy_head_loss": 389.9279, "value_head_loss": 0.2066, "loss": 390.1349},
    {"policy_head_loss": 371.3614, "value_head_loss": 0.2067, "loss": 371.5681},
    {"policy_head_loss": 373.2598, "value_head_loss": 0.2067, "loss": 373.4667},
    {"policy_head_loss": 357.5609, "value_head_loss": 0.2065, "loss": 357.7670},
    {"policy_head_loss": 353.7310, "value_head_loss": 0.2066, "loss": 353.9374},
    {"policy_head_loss": 326.0812, "value_head_loss": 0.2066, "loss": 326.2877},
    {"policy_head_loss": 320.7809, "value_head_loss": 0.2064, "loss": 320.9875},
    {"policy_head_loss": 304.6947, "value_head_loss": 0.2066, "loss": 304.9014},
    {"policy_head_loss": 302.4235, "value_head_loss": 0.2069, "loss": 302.6302},
    {"policy_head_loss": 282.2320, "value_head_loss": 0.2064, "loss": 282.4386},
    # Training 6 (Main):
    {"policy_head_loss": 325.1394, "value_head_loss": 0.2025, "loss": 325.3422},
    {"policy_head_loss": 299.9349, "value_head_loss": 0.2025, "loss": 300.1375},
    {"policy_head_loss": 304.1692, "value_head_loss": 0.2023, "loss": 304.3719},
    {"policy_head_loss": 293.5392, "value_head_loss": 0.2025, "loss": 293.7417},
    {"policy_head_loss": 275.0337, "value_head_loss": 0.2024, "loss": 275.2359},
    {"policy_head_loss": 273.6742, "value_head_loss": 0.2025, "loss": 273.8768},
    {"policy_head_loss": 264.4348, "value_head_loss": 0.2025, "loss": 264.6374},
    {"policy_head_loss": 250.4412, "value_head_loss": 0.2024, "loss": 250.6436},
    {"policy_head_loss": 241.1968, "value_head_loss": 0.2022, "loss": 241.3991},
    {"policy_head_loss": 234.0853, "value_head_loss": 0.2024, "loss": 234.2877},
    # Training 7 (Main):
    {"policy_head_loss": 228.0964, "value_head_loss": 0.1913, "loss": 228.2877},
    {"policy_head_loss": 214.9679, "value_head_loss": 0.1911, "loss": 215.1589},
    {"policy_head_loss": 201.9315, "value_head_loss": 0.1911, "loss": 202.1225},
    {"policy_head_loss": 193.8684, "value_head_loss": 0.1912, "loss": 194.0598},
    {"policy_head_loss": 188.9584, "value_head_loss": 0.1910, "loss": 189.1496},
    {"policy_head_loss": 187.4357, "value_head_loss": 0.1910, "loss": 187.6268},
    {"policy_head_loss": 179.7856, "value_head_loss": 0.1912, "loss": 179.9768},
    {"policy_head_loss": 163.0483, "value_head_loss": 0.1912, "loss": 163.2394},
    {"policy_head_loss": 163.9782, "value_head_loss": 0.1912, "loss": 164.1695},
    {"policy_head_loss": 159.1150, "value_head_loss": 0.1910, "loss": 159.3061},
    # Training 8 (Main):
    {"policy_head_loss": 145.7193, "value_head_loss": 0.1149, "loss": 145.8341},
    {"policy_head_loss": 138.4205, "value_head_loss": 0.1148, "loss": 138.5354},
    {"policy_head_loss": 133.2272, "value_head_loss": 0.1149, "loss": 133.3422},
    {"policy_head_loss": 141.6408, "value_head_loss": 0.1149, "loss": 141.7558},
    {"policy_head_loss": 131.8368, "value_head_loss": 0.1149, "loss": 131.9518},
    {"policy_head_loss": 131.4586, "value_head_loss": 0.1148, "loss": 131.5733},
    {"policy_head_loss": 129.2348, "value_head_loss": 0.1149, "loss": 129.3496},
    {"policy_head_loss": 116.5135, "value_head_loss": 0.1149, "loss": 116.6284},
    {"policy_head_loss": 110.9446, "value_head_loss": 0.1148, "loss": 111.0594},
    {"policy_head_loss": 109.9295, "value_head_loss": 0.1148, "loss": 110.0444},
    # Training 9 (Main):
    {"policy_head_loss": 114.9834, "value_head_loss": 0.1228, "loss": 115.1063},
    {"policy_head_loss": 109.4365, "value_head_loss": 0.1228, "loss": 109.5593},
    {"policy_head_loss": 112.0622, "value_head_loss": 0.1228, "loss": 112.1850},
    {"policy_head_loss": 107.9688, "value_head_loss": 0.1228, "loss": 108.0915},
    {"policy_head_loss": 110.6468, "value_head_loss": 0.1228, "loss": 110.7697},
    {"policy_head_loss": 100.2440, "value_head_loss": 0.1228, "loss": 100.3667},
    {"policy_head_loss": 93.2542, "value_head_loss": 0.1228, "loss": 93.3770},
    {"policy_head_loss": 83.0821, "value_head_loss": 0.1228, "loss": 83.2049},
    {"policy_head_loss": 94.0255, "value_head_loss": 0.1228, "loss": 94.1482},
    {"policy_head_loss": 87.3940, "value_head_loss": 0.1229, "loss": 87.5167},
    # Training 10 (Main):
    {"policy_head_loss": 94.8622, "value_head_loss": 0.1348, "loss": 94.9971},
    {"policy_head_loss": 88.3867, "value_head_loss": 0.1347, "loss": 88.5213},
    {"policy_head_loss": 91.7828, "value_head_loss": 0.1348, "loss": 91.9176},
    {"policy_head_loss": 91.0895, "value_head_loss": 0.1348, "loss": 91.2243},
    {"policy_head_loss": 89.2028, "value_head_loss": 0.1347, "loss": 89.3375},
    {"policy_head_loss": 81.9306, "value_head_loss": 0.1347, "loss": 82.0654},
    {"policy_head_loss": 74.2971, "value_head_loss": 0.1347, "loss": 74.4318},
    {"policy_head_loss": 78.9353, "value_head_loss": 0.1347, "loss": 79.0701},
    {"policy_head_loss": 72.3736, "value_head_loss": 0.1347, "loss": 72.5083},
    {"policy_head_loss": 70.2172, "value_head_loss": 0.1347, "loss": 70.3519},

    # Training 1 (Fine):
    {"policy_head_loss": 49.8315, "value_head_loss": 0.0307, "loss": 49.8621},
    {"policy_head_loss": 34.1236, "value_head_loss": 0.0307, "loss": 34.1543},
    {"policy_head_loss": 54.3056, "value_head_loss": 0.0307, "loss": 54.3363},
    {"policy_head_loss": 55.9112, "value_head_loss": 0.0306, "loss": 55.9419},
    {"policy_head_loss": 60.2396, "value_head_loss": 0.0307, "loss": 60.2702},
    {"policy_head_loss": 41.6282, "value_head_loss": 0.0307, "loss": 41.6589},
    {"policy_head_loss": 53.4272, "value_head_loss": 0.0307, "loss": 53.4578},
    {"policy_head_loss": 58.0263, "value_head_loss": 0.0307, "loss": 58.0570},
    {"policy_head_loss": 28.9029, "value_head_loss": 0.0306, "loss": 28.9335},
    {"policy_head_loss": 6.4644, "value_head_loss": 0.0307, "loss": 6.4951},
    {"policy_head_loss": 6.4320, "value_head_loss": 0.0306, "loss": 6.4626},
    {"policy_head_loss": 75.2329, "value_head_loss": 0.0307, "loss": 75.2635},
    {"policy_head_loss": 59.6781, "value_head_loss": 0.0307, "loss": 59.7088},
    {"policy_head_loss": 48.0163, "value_head_loss": 0.0307, "loss": 48.0470},
    {"policy_head_loss": 43.2193, "value_head_loss": 0.0307, "loss": 43.2500},
    {"policy_head_loss": 17.1115, "value_head_loss": 0.0307, "loss": 17.1422},
    {"policy_head_loss": 5.3295, "value_head_loss": 0.0306, "loss": 5.3602},
    {"policy_head_loss": 39.4111, "value_head_loss": 0.0306, "loss": 39.4417},
    {"policy_head_loss": 39.5743, "value_head_loss": 0.0306, "loss": 39.6050},
    {"policy_head_loss": 5.1296, "value_head_loss": 0.0306, "loss": 5.1603},

    # Training 2 (Fine):
    {"policy_head_loss": 45.0195, "value_head_loss": 0.1119, "loss": 45.1313},
    {"policy_head_loss": 37.5489, "value_head_loss": 0.1119, "loss": 37.6608},
    {"policy_head_loss": 42.7995, "value_head_loss": 0.1119, "loss": 42.9114},
    {"policy_head_loss": 40.9976, "value_head_loss": 0.1119, "loss": 41.1095},
    {"policy_head_loss": 40.4061, "value_head_loss": 0.1119, "loss": 40.5181},
    {"policy_head_loss": 41.3574, "value_head_loss": 0.1119, "loss": 41.4693},
    {"policy_head_loss": 37.9304, "value_head_loss": 0.1119, "loss": 38.0423},
    {"policy_head_loss": 26.2906, "value_head_loss": 0.1119, "loss": 26.4025},
    {"policy_head_loss": 31.6421, "value_head_loss": 0.1119, "loss": 31.7540},
    {"policy_head_loss": 39.2155, "value_head_loss": 0.1119, "loss": 39.3273},
    {"policy_head_loss": 27.0303, "value_head_loss": 0.1119, "loss": 27.1422},
    {"policy_head_loss": 32.4028, "value_head_loss": 0.1119, "loss": 32.5147},
    {"policy_head_loss": 25.4008, "value_head_loss": 0.1119, "loss": 25.5127},
    {"policy_head_loss": 37.3029, "value_head_loss": 0.1119, "loss": 37.4148},
    {"policy_head_loss": 28.2665, "value_head_loss": 0.1119, "loss": 28.3784},
    {"policy_head_loss": 22.0301, "value_head_loss": 0.1119, "loss": 22.1420},
    {"policy_head_loss": 27.8228, "value_head_loss": 0.1119, "loss": 27.9347},
    {"policy_head_loss": 17.8012, "value_head_loss": 0.1119, "loss": 17.9131},
    {"policy_head_loss": 34.5442, "value_head_loss": 0.1119, "loss": 34.6561},
    {"policy_head_loss": 25.2719, "value_head_loss": 0.1119, "loss": 25.3838},
]

def plot_loss_separated(loss_data):
    epochs = range(1, len(loss_data) + 1)
    policy_losses = [epoch["policy_head_loss"] for epoch in loss_data]
    value_losses = [epoch["value_head_loss"] for epoch in loss_data]
    total_losses = [epoch["loss"] for epoch in loss_data]

    # Plot Policy Head Loss and Total Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, policy_losses, label="Policy Head Loss", marker='o')
    #plt.plot(epochs, total_losses, label="Total Loss", marker='o')

    plt.axvspan(1, 100, color='lightgreen', alpha=0.3)
    plt.axvspan(100, 150, color='lightcoral', alpha=0.3)

    plt.axvline(x=1, color='gray', linestyle='--', linewidth=1)
    plt.axvline(x=100, color='gray', linestyle='--', linewidth=1)
    plt.text(1, (min(policy_losses) + max(policy_losses)) / 2, "Main", color="gray", fontsize=10, ha="left", va="center",rotation=90)
    plt.text(100, (min(policy_losses) + max(policy_losses)) / 2, "Fine Tuning", color="gray", fontsize=10, ha="left", va="center",rotation=90)

    plt.xlabel("Epoche", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Policy Head Loss über die Epochen", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Plot Value Head Loss separately
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, value_losses, label="Value Head Loss", marker='o', color='purple')

    plt.axvspan(1, 100, color='lightgreen', alpha=0.3)
    plt.axvspan(100, 150, color='lightcoral', alpha=0.3)

    plt.axvline(x=1, color='gray', linestyle='--', linewidth=1)
    plt.axvline(x=100, color='gray', linestyle='--', linewidth=1)
    plt.text(1, (min(value_losses) + max(value_losses)) / 2, "Main", color="gray", fontsize=10, ha="left", va="center",rotation=90)
    plt.text(100, (min(value_losses) + max(value_losses)) / 2, "Fine Tuning", color="gray", fontsize=10, ha="left", va="center",rotation=90)

    plt.xlabel("Epoche", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Value Head Loss über die Epochen", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Call the function to plot the losses
plot_loss_separated(loss_data)
