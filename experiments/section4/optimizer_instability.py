import torch
import random


class RandomLabelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.randomize_labels()

    def randomize_labels(self):
        self.labels = [
            random.randint(0, 9)
            for _ in range(len(self.dataset))
        ]

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]

        return x, self.labels[idx]

    def __len__(self):
        return len(self.dataset)



def train_instability_experiment(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    reshuffle_every=5000,
    total_steps=50000,
):
    model.train()

    step = 0

    while step < total_steps:
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)

            loss = criterion(logits, y)

            loss.backward()

            optimizer.step()

            if step % reshuffle_every == 0:
                dataloader.dataset.randomize_labels()

                print(
                    f"Reshuffled labels at step {step}"
                )

            if step % 100 == 0:
                print(
                    f"Step {step} | Loss {loss.item():.4f}"
                )

            step += 1

            if step >= total_steps:
                break