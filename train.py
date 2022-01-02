from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader
import torch.nn as nn


def train(model, data_sets, optimizer, num_epochs: int, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loaders = {"train": DataLoader(data_sets["train"], batch_size=batch_size, shuffle=True),
                    "test": DataLoader(data_sets["test"], batch_size=batch_size, shuffle=False)}
    model.to(device)

    best_acc = 0.0
    best_f1 = 0.0

    for epoch in range(num_epochs):
        # print(f'Epoch {epoch + 1}/{num_epochs}')
        # print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_acc = 0.0

            pred_tensor = torch.empty(size=(16,), dtype=torch.int32, device=device)
            true_tensor = torch.empty(size=(16,), dtype=torch.int32, device=device)
            i = 0
            for batch in data_loaders[phase]:
                batch_size = 0
                for k, v in batch.items():
                    batch[k] = v.to(device)
                    batch_size = v.shape[0]

                optimizer.zero_grad()
                if phase == 'train':
                    outputs, loss = model(**batch)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1000)
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs, loss = model(**batch)
                pred = outputs.argmax(dim=-1).clone().detach().cpu()
                cur_num_correct = accuracy_score(batch['labels'].cpu().view(-1), pred.view(-1), normalize=False)
                if i == 0:
                    pred_tensor = pred
                    true_tensor = batch['labels'].cpu().view(-1)
                else:
                    pred_tensor = torch.cat((pred_tensor, pred.view(-1)))
                    true_tensor = torch.cat((true_tensor, batch['labels'].cpu().view(-1)))
                i += 1

                running_loss += loss.item() * batch_size
                running_acc += cur_num_correct

            f1_sc = f1_score(true_tensor, pred_tensor)

            epoch_loss = running_loss / len(data_sets[phase])
            epoch_acc = running_acc / len(data_sets[phase])

            epoch_acc = round(epoch_acc, 5)
            # if phase.title() == "test":
            #     print(f'{phase.title()} Loss: {epoch_loss:.4e} Accuracy: {epoch_acc}')
            #     print(f"{phase.title()} f1 score for epoch {epoch} is {f1_sc}")
            # else:
            #     print(f'{phase.title()} Loss: {epoch_loss:.4e} Accuracy: {epoch_acc}')
            #     print(f"{phase.title()} f1 score for epoch {epoch} is {f1_sc}")
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
            if phase == 'test' and f1_sc > best_f1:
                best_f1 = f1_sc
    print()

    # print(f'Best Validation Accuracy: {best_acc:4f}')
    return model, best_f1
