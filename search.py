from validate import *


save_path = os.path.join(os.getcwd(), "data", "checkpoints/")

highest_cp = -1
for _, _, files in os.walk(save_path):
    for filename in files:
        cp = int(re.split("[-.]", filename)[-2])
        if cp > highest_cp:
            highest_cp = cp

num_classes = 50
batch_size = 64
print("num_classes: ", num_classes, "batch_size", batch_size)

model = JigsawNet(n_classes=num_classes).to(device)

validation_data = np.load(f"data/preprocessed_validation.npy")
validation_data = torch.from_numpy(validation_data).float()

validation_dataset = JigsawValidationDataset(validation_data)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

refs = [
    "/mnt/c/Users/dongh/Downloads/result_22/validation.txt",
    "/mnt/c/Users/dongh/Downloads/result_23/validation.txt",
    "/mnt/c/Users/dongh/Downloads/result_24/validation.txt",
]


while checkpoint <= highest_cp:
    print("checkpoint: ", checkpoint, "highest_cp", highest_cp)

    reset_random_generators()
    evaluate_model(model, validation_loader, checkpoint=checkpoint)

    list_a = np.loadtxt(f"data/validation.txt")

    for ref in refs:
        list_b = np.loadtxt(ref)

        diff = {}
        count = 0
        for i in range(len(list_b)):
            if list_a[i] != list_b[i]:
                # print(f"found difference: {i}")
                diff[i] = (list_a[i], list_b[i])
                count += 1

        # print("#Difference :", len(diff), count)

        if len(diff) == 0:
            print("Found match:", checkpoint, ref)

    checkpoint += 1
